from re import L
import torch as th

def token_entropy_loss(student_attn_dict, teacher_attn_dict):
    """
        Entropy loss is to minimize the difference between the entropy of original token-level
        attention logits distribution and merged token-level attention logits distribution.

        Args:
            student_attn_dict: {1: attention_logits, 2: attention_logits, 3: attention_logits.....}
            teacher_attn_dict: {1: attention_logits, 2: attention_logits, 3: attention_logits.....}
            # attention_logits.shape = (B,L,L)
        Returns: batch_entropy_loss: (B,)
    """
    shape = None
    for layer in student_attn_dict.keys():
        shape = student_attn_dict[layer].shape
        break
    B,L,_ = shape
    batch_entropy_loss = th.zeros(th.zeros((B,))).to(student_attn_dict[1].device)
    for layer in student_attn_dict.keys():
        s_entropy = th.log(1 + th.var(student_attn_dict[layer], dim=-1))
        t_entropy = th.log(1 + th.var(teacher_attn_dict[layer], dim=-1))
        batch_entropy_loss += th.mean((t_entropy - s_entropy)**2, dim=-1)

    return batch_entropy_loss


class EasyDict:
    def __init__(self, sub_dict):
        for k, v in sub_dict.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return th.mean(x, dim=list(range(1, len(x.size()))))


def log_state(state):
    result = []

    sorted_state = dict(sorted(state.items()))
    for key, value in sorted_state.items():
        # Check if the value is an instance of a class
        if "<object" in str(value) or "object at" in str(value):
            result.append(f"{key}: [{value.__class__.__name__}]")
        else:
            result.append(f"{key}: {value}")

    return "\n".join(result)

