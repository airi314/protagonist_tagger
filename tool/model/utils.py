def load_model(library, ner_model_dir_path, save_personal_titles):
    if library == 'spacy':
        from tool.model.spacy_model import SpacyModel
        if ner_model_dir_path == 'default':
            ner_model_dir_path = 'en_core_web_sm'
        model = SpacyModel(ner_model_dir_path, save_personal_titles)

    elif library == 'nltk':
        from tool.model.nltk_model import NLTKModel
        model = NLTKModel(save_personal_titles)

    elif library == 'stanza':
        from tool.model.stanza_model import StanzaModel
        model = StanzaModel(save_personal_titles)

    elif library == 'flair':
        from tool.model.flair_model import FlairModel
        if ner_model_dir_path == 'default':
            ner_model_dir_path = 'ner'
        model = FlairModel(ner_model_dir_path, save_personal_titles)

    else:
        raise Exception('Library "' + library + '" is not supported. You can choose one of: spacy, nltk, stanza and '
                                                'flair.')
    return model
