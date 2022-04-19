import math

import torch.nn.functional as F
import torch

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('cross_entropy_agg')
class CrossEntropyAGGCriterion(FairseqCriterion):
    """AGG loss based on cross entropy loss class of FairSeq"""
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.count_memory = None
        self.word_count = None
        if self.task.args.task == 'language_modeling':
            vocab_size = len(self.task.dictionary)
        elif self.task.args.task == 'translation':
            vocab_size = len(self.task.tgt_dict)
        if task.args.cpu:
            self.token_nll = -torch.ones(vocab_size)
        else:
            self.token_nll = -torch.ones(vocab_size, device="cuda")

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def set_word_count(self, itr_per_epoch):
        if self.task.args.task == 'language_modeling':
            vocab_size = len(self.task.dictionary)
        elif self.task.args.task == 'translation':
            vocab_size = len(self.task.tgt_dict)
        self.count_memory = itr_per_epoch
        if self.task.args.cpu:
            self.word_count = torch.zeros([self.count_memory + 1, vocab_size])
        else:
            self.word_count = torch.zeros([self.count_memory + 1, vocab_size], device="cuda")

    def forward(self, model, sample, reduce=True, itr=1):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'], features_only=True)
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce, itr=itr)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True, itr=1):
        epsilon = self.task.args.label_smoothing
        alpha = self.task.args.agg_alpha

        features = net_output[0]
        target = model.get_targets(sample, net_output).view(-1)
        nspecial = model.decoder.dictionary.nspecial

        # update token counter memory with current mini-batch samples
        count_mem_idx = itr % (self.count_memory + 1)
        self.word_count[count_mem_idx] = 0
        self.word_count[count_mem_idx - 1, target] += (target.unsqueeze(1) == target).float().sum(1)

        # get token appearance
        word_count = self.word_count.sum(0)

        # grouping rare tokens according to the token appearance in the current step
        f1 = word_count / min(itr, self.count_memory)
        f1[:nspecial] = 1.
        rare_group_mask = (f1 < alpha)
        rare_group_indices = rare_group_mask.float().nonzero().squeeze(-1)
        non_rare_group_indices = (1 - rare_group_mask.float()).nonzero().squeeze(-1)
        f1[non_rare_group_indices] = 1.

        # calculate normalized frequency of very-rare tokens
        c_r_mean = word_count[rare_group_indices].mean()
        f2 = word_count / (c_r_mean + 1e-7)  # abs_p_rare
        f2[:nspecial] = 1.
        f2[non_rare_group_indices] = 1.
        f2 = f2.clip(max=1.)

        output_weight = model.decoder.output_projection.weight * 2 / 2
        target_mask = (target.unsqueeze(1) == rare_group_indices).float().sum(1)
        rare_tgt_idx = target_mask.nonzero().squeeze(-1)
        non_rare_tgt_idx = (1. - target_mask).nonzero().squeeze(-1)

        # calculate gate1 and gate2 for gating gradients of the token embedding vectors
        gate1 = f1.expand(non_rare_tgt_idx.size(0), output_weight.size(0)).clone().detach()
        gate1[torch.arange(non_rare_tgt_idx.size(0)), target[non_rare_tgt_idx]] = 1.
        gate2 = f2.expand(rare_tgt_idx.size(0), output_weight.size(0)).clone().detach()
        gate2[torch.arange(rare_tgt_idx.size(0)), target[rare_tgt_idx]] = 1.

        # calculate the gated logits for gate1
        features_size = features.size()
        features_nonrare = features.detach().contiguous().view(-1, features_size[2])[non_rare_tgt_idx]
        logits_nonrare = torch.nn.functional.linear(features_nonrare, output_weight)
        logits_size = logits_nonrare.size()
        logits_nonrare = logits_nonrare.view(-1, logits_size[-1])
        logits_nonrare_gated = gate1 * logits_nonrare + (1.-gate1) * logits_nonrare.detach()

        # calculate the gated logits for gate2
        features_rare = features.detach().contiguous().view(-1, features_size[2])[rare_tgt_idx]
        logits_rare = torch.nn.functional.linear(features_rare, output_weight)
        logits_rare = logits_rare.view(-1, logits_size[-1])
        logits_rare_gated = gate2 * logits_rare + (1.-gate2) * logits_rare.detach()

        # calculate the original nll logits for gradients about feature vectors
        logits_feature = torch.nn.functional.linear(features, output_weight.detach())
        logits_feature = logits_feature.view(-1, logits_size[-1])

        # calculate the log-probabilities of logits
        lprobs_nonrare = lprobs_nonrare_ls = F.log_softmax(logits_nonrare_gated.float(), dim=-1).view(-1,
                                                                                                      logits_size[-1])
        lprobs_rare = lprobs_rare_ls = F.log_softmax(logits_rare_gated.float(), dim=-1).view(-1, logits_size[-1])
        lprobs_feature = lprobs_feature_ls = F.log_softmax(logits_feature.float(), dim=-1).view(-1, logits_size[-1])

        # calculate AGG loss for language modeling task
        if self.task.args.task == 'language_modeling':
            if target.dim() == lprobs_nonrare.dim() - 1:
                target = target.unsqueeze(-1)
            nll_loss_nr_emb = -lprobs_nonrare.gather(dim=-1, index=target[non_rare_tgt_idx])
            nll_loss_r_emb = -lprobs_rare.gather(dim=-1, index=target[rare_tgt_idx])
            nll_loss_feature = -lprobs_feature.gather(dim=-1, index=target)

            if self.padding_idx is not None:
                pad_mask1 = target[non_rare_tgt_idx].eq(self.padding_idx)
                pad_mask2 = target[rare_tgt_idx].eq(self.padding_idx)
                pad_mask = target.eq(self.padding_idx)
                nll_loss_nr_emb.masked_fill_(pad_mask1, 0.)
                nll_loss_r_emb.masked_fill_(pad_mask2, 0.)
                nll_loss_feature.masked_fill(pad_mask, 0.)
            else:
                nll_loss_nr_emb = nll_loss_nr_emb.squeeze(-1)
                nll_loss_r_emb = nll_loss_r_emb.squeeze(-1)
                nll_loss_feature = nll_loss_feature.squeeze(-1)

            if reduce:
                nll_loss_nr_emb = nll_loss_nr_emb.sum()
                nll_loss_r_emb = nll_loss_r_emb.sum()
                nll_loss_feature = nll_loss_feature.sum()
            loss = nll_loss_feature + nll_loss_nr_emb + nll_loss_r_emb

        # calculate AGG loss with label smoothing for machine translation task
        if self.task.args.task == 'translation':
            if target.dim() == lprobs_nonrare.dim() - 1:
                target = target.unsqueeze(-1)

            nll_loss_nr_emb = -lprobs_nonrare.gather(dim=-1, index=target[non_rare_tgt_idx])
            nll_loss_r_emb = -lprobs_rare.gather(dim=-1, index=target[rare_tgt_idx])
            nll_loss_feature = -lprobs_feature.gather(dim=-1, index=target)

            smooth_loss_nr_emb = -lprobs_nonrare_ls.sum(dim=-1, keepdim=True)
            smooth_loss_r_emb = -lprobs_rare_ls.sum(dim=-1, keepdim=True)
            smooth_loss_feature = -lprobs_feature_ls.sum(dim=-1, keepdim=True)

            if self.padding_idx is not None:
                pad_mask1 = target[non_rare_tgt_idx].eq(self.padding_idx)
                pad_mask2 = target[rare_tgt_idx].eq(self.padding_idx)
                pad_mask = target.eq(self.padding_idx)
                nll_loss_nr_emb.masked_fill_(pad_mask1, 0.)
                nll_loss_r_emb.masked_fill_(pad_mask2, 0.)
                nll_loss_feature.masked_fill(pad_mask, 0.)
                smooth_loss_nr_emb.masked_fill_(pad_mask1, 0.)
                smooth_loss_r_emb.masked_fill_(pad_mask2, 0.)
                smooth_loss_feature.masked_fill(pad_mask, 0.)
            else:
                nll_loss_nr_emb = nll_loss_nr_emb.squeeze(-1)
                nll_loss_r_emb = nll_loss_r_emb.squeeze(-1)
                nll_loss_feature = nll_loss_feature.squeeze(-1)
                smooth_loss_nr_emb = smooth_loss_nr_emb.squeeze(-1)
                smooth_loss_r_emb = smooth_loss_r_emb.squeeze(-1)
                smooth_loss_feature = smooth_loss_feature.squeeze(-1)

            if reduce:
                nll_loss_nr_emb = nll_loss_nr_emb.sum()
                nll_loss_r_emb = nll_loss_r_emb.sum()
                nll_loss_feature = nll_loss_feature.sum()
                smooth_loss_nr_emb = smooth_loss_nr_emb.sum()
                smooth_loss_r_emb = smooth_loss_r_emb.sum()
                smooth_loss_feature = smooth_loss_feature.sum()

            eps_i = epsilon / lprobs_nonrare.size(-1)
            loss_nr_emb = (1. - epsilon) * nll_loss_nr_emb + eps_i * smooth_loss_nr_emb
            loss_r_emb = (1. - epsilon) * nll_loss_r_emb + eps_i * smooth_loss_r_emb
            loss_feature = (1. - epsilon) * nll_loss_feature + eps_i * smooth_loss_feature

            loss = loss_nr_emb + loss_r_emb + loss_feature

        return loss, nll_loss_feature.detach()

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        else:
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
