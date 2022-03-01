def get_litbank_text_parts(path):
    with open(path) as f:
        text = f.read()
    text_parts = text.split('\n\n\n')
    text_parts = ['\n\n'.join([' '.join(paragraph.split('\n')) for paragraph in part.split('\n\n')])
        for part in text_parts]
    return text_parts

def get_litbank_text(path):
    text_parts = get_litbank_text_parts(path)
    text = '\n\n\n'.join(text_parts)
    return text
