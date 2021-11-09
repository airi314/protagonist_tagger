def load_model(library, ner_model_dir_path):
    if library == 'spacy':
        from tool.model.spacy_model import SpacyModel
        if ner_model_dir_path is None:
            ner_model_dir_path = 'en_core_web_sm'
        model = SpacyModel(ner_model_dir_path)

    elif library == 'nltk':
        from tool.model.nltk_model import NLTKModel
        model = NLTKModel()

    elif library == 'stanza':
        from tool.model.stanza_model import StanzaModel
        model = StanzaModel()

    elif library == 'flair':
        from tool.model.flair_model import FlairModel
        if ner_model_dir_path is None:
            ner_model_dir_path = 'ner'
        model = FlairModel(ner_model_dir_path)

    else:
        raise Exception('Library "' + library + '" is not supported. You can choose one of: spacy, nltk, stanza and '
                                                'flair.')
    return model
