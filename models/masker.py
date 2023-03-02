from transformers import pipeline


unmasker = pipeline('fill-mask', model='bert-base-uncased')


def masker_result(question_tokens, result):
    sequence_start = unmasker(f'[MASK] {question_tokens}')[0]['sequence']
    sequence_end = unmasker(f'{sequence_start} [MASK] ')[0]['sequence']
    result_sequence_mask = unmasker(f'{result} [MASK]')[0]
    sequence_mask_result = unmasker(f'{sequence_end} [MASK] {result_sequence_mask["sequence"]}')[0]['token_str']
    sequence_result = f'{sequence_end} {sequence_mask_result} {result} {result_sequence_mask["token_str"]}'
    return sequence_result


def masker_result_raw(question_tokens, result):
    sequence_start_mask = unmasker(f'[MASK] {question_tokens}')[0]
    sequence_start_token = sequence_start_mask['token_str']
    sequence_start = sequence_start_mask['sequence']
    sequence_middle_mask = unmasker(f'{sequence_start} [MASK] {result}')[0]
    sequence_middle_token = sequence_middle_mask['token_str']
    sequence_end_mask = unmasker(f'{sequence_middle_mask["sequence"]} {result} [MASK] ')[0]['token_str']
    sequence_result = f'{sequence_start_token} {question_tokens} {sequence_middle_token} {result} {sequence_end_mask}'
    return sequence_result
