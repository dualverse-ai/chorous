"""Prompt construction from nanochat CORE evaluation (MIT; see licenses/)."""
from jinja2 import Template

def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a multiple choice question"""
    template_str = '\n{%- for example in fewshot_examples -%}\n{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}\n\n{% endfor -%}\n{{ item.query }}{{ continuation_delimiter }}{{ choice }}'.strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {'fewshot_examples': fewshot_examples, 'continuation_delimiter': continuation_delimiter, 'item': item}
    prompts = [template.render(choice=choice, **context) for choice in item['choices']]
    return prompts

def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a schema question"""
    template_str = '\n{%- for example in fewshot_examples -%}\n{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}\n\n{% endfor -%}\n{{ context }}{{ continuation_delimiter }}{{ item.continuation }}'.strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {'fewshot_examples': fewshot_examples, 'continuation_delimiter': continuation_delimiter, 'item': item}
    prompts = [template.render(context=context_option, **context) for context_option in item['context_options']]
    return prompts

def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    """
    Render complete prompt for a language modeling task.
    Notice that we manually trim the context in the template,
    which in some datasets seems to have trailing whitespace (which we don't want).
    """
    template_str = '\n{%- for example in fewshot_examples -%}\n{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}\n\n{% endfor -%}\n{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}'.strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {'fewshot_examples': fewshot_examples, 'continuation_delimiter': continuation_delimiter, 'item': item}
    prompt_without = template.render(include_continuation=False, **context)
    prompt_with = template.render(include_continuation=True, **context)
    prompt_without = prompt_without.strip()
    return [prompt_without, prompt_with]

def find_common_length(token_sequences, direction='left'):
    """
    Find the length of the common prefix or suffix across token sequences
    - direction: 'left' for prefix, 'right' for suffix
    """
    min_len = min((len(seq) for seq in token_sequences))
    indices = {'left': range(min_len), 'right': range(-1, -min_len - 1, -1)}[direction]
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all((seq[idx] == token for seq in token_sequences)):
            return i
    return min_len

def batch_sequences_mc(tokenizer, prompts):
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    answer_start_idx = find_common_length(tokens, direction='left')
    start_indices = [answer_start_idx] * len(prompts)
    end_indices = [len(x) for x in tokens]
    return (tokens, start_indices, end_indices)

def batch_sequences_schema(tokenizer, prompts):
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    suffix_length = find_common_length(tokens, direction='right')
    end_indices = [len(x) for x in tokens]
    start_indices = [ei - suffix_length for ei in end_indices]
    return (tokens, start_indices, end_indices)

def batch_sequences_lm(tokenizer, prompts):
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    tokens_without, tokens_with = tokens
    start_idx, end_idx = (len(tokens_without), len(tokens_with))
    assert start_idx < end_idx, 'prompt without is supposed to be a prefix of prompt with'
    assert tokens_without == tokens_with[:start_idx], 'prompt without is supposed to be a prefix of prompt with'
    return ([tokens_with], [start_idx], [end_idx])
