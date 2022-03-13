from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

import numpy as np

from tool.model.ner_model import NERModel


class TransformerModel(NERModel):

    def __init__(self, model_path, save_personal_titles, fix_personal_titles):
        super().__init__(save_personal_titles, fix_personal_titles)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model = pipeline("token-classification", aggregation_strategy="simple", model=model, tokenizer=tokenizer)
        print('Transformers model loaded.')

    def get_doc_entities(self, text):
        results = self.model(text)
        # inputs_with_offsets = self.tokenizer(text, return_offsets_mapping=True)
        entities = []
        for index, ent in enumerate(results):
            if ent['entity_group'] == "PER":
                start, end = ent['start'], ent['end']
                ent_text = ent['word']
                if self.fix_personal_titles and ent_text.startswith(self.personal_titles):
                    start += (1 + len(ent_text.split(' ')[0]))
                # if self.save_personal_titles:
                #     personal_title = self.recognize_personal_title(inputs_with_offsets, ent)
                #     entities.append([start, end, "PERSON", personal_title])
                # else:
                #     entities.append([start, end, "PERSON"])

        return {'content': text, 'entities': entities}

    def recognize_personal_title(self, inputs_with_offsets, ent):
        personal_title = None
        offset_mapping = np.array(inputs_with_offsets['offset_mapping'])
        if ent['start'] > 0:
            token_index = np.where(offset_mapping == ent['start'])[0][0]
            word_before_name = inputs_with_offsets.tokens()[token_index-1]
            if word_before_name == '.':
                word_before_name = inputs_with_offsets.tokens()[token_index-2] + '.'
            if word_before_name.replace(".", "") in self.personal_titles:
                personal_title = word_before_name.replace(".", "")
            if word_before_name.lower() == "the":
                personal_title = "the"

        return personal_title
