from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

from tool.model.ner_model import NERModel


class TransformerModel(NERModel):

    def __init__(self, model_path, save_personal_titles, fix_personal_titles):
        super().__init__(save_personal_titles, fix_personal_titles)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model = pipeline(
            "token-classification",
            aggregation_strategy="simple",
            model=model,
            tokenizer=self.tokenizer)
        self.logger.info('Transformers model loaded.')

    def get_doc_entities(self, text):
        results = self.model(text)
        tokens = self.tokenizer(text)
        entities = []
        for index, ent in enumerate(results):
            if ent['entity_group'] == "PER":
                start, end = ent['start'], ent['end']
                start, end = self.fix_entity(start, end, tokens)
                ent_text = text[start:end]
                if self.fix_personal_titles and ent_text.startswith(
                        self.personal_titles):
                    start += (1 + len(ent_text.split(' ')[0]))
                if self.save_personal_titles:
                    personal_title = self.recognize_personal_title(
                        ent, text, tokens)
                    if [start, end, "PERSON", personal_title] not in entities:
                        entities.append([start, end, "PERSON", personal_title])
                elif not [start, end, "PERSON"] in entities:
                    entities.append([start, end, "PERSON"])

        entities_merged = []
        i = 0
        while i < len(entities):
            if i < len(entities) - \
                    1 and entities[i][1] + 1 == entities[i + 1][0]:
                entities_merged.append(
                    [entities[i][0], entities[i + 1][1], "PERSON"])
                i += 2
            else:
                entities_merged.append(entities[i])
                i += 1
        return text, entities_merged

    def recognize_personal_title(self, ent, text, tokens):
        personal_title = None
        if ent['start'] > 0:
            token_index = tokens.char_to_word(ent[0])
            previous_token = tokens.word_to_chars(token_index - 1)
            word_before_name = text[previous_token.start:previous_token.end]
            if word_before_name == '.':
                previous_token = tokens.word_to_chars(token_index - 1)
                word_before_name = text[previous_token.start:previous_token.end] + '.'
            if word_before_name in self.personal_titles:
                personal_title = word_before_name
        if personal_title:
            print(ent, text, personal_title)
        return personal_title

    def fix_entity(self, start, end, tokens):
        # print('tokens', tokens)
        # print(start, end)
        word_start = tokens.char_to_word(start)
        # if not word_start:
        #     word_start = tokens.char_to_word(start + 1)
        word_end = tokens.char_to_word(end - 1)
        # print('word_end', word_end)
        # if not word_end:
        #     word_end = tokens.char_to_word(end - 1)
        #     end = tokens.word_to_chars(word_end).end
        #     print('end1', end)
        # else:
        # print('tokens.word_to_chars(word_end)', tokens.word_to_chars(word_end))
        end = tokens.word_to_chars(word_end).end
        # print('end2', end)
        start = tokens.word_to_chars(word_start).start
        # print('start', start)
        return start, end
