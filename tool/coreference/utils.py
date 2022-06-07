def load_model(library, model_name):
    if library == 'coreferee':
        from tool.coreference.coreferee import Coreferee
        if model_name is None:
            model_name = 'en_core_web_sm'
        model = Coreferee(model_name)

    elif library == 'fast_coref':
        from tool.coreference.fast_coref import FastCoref
        if model_name is None:
            model_name = 'ontonotes_best'
        model = FastCoref(model_name)

    elif library == 'long_doc_coref':
        from tool.coreference.long_doc_coref import LongDocCoref
        if model_name is None:
            model_name = 'ontonotes'
        model = LongDocCoref(model_name)

    elif library == 'wl_coref':
        from tool.coreference.wl_coref import WLCoref
        if model_name is None:
            model_name = "roberta"
        model = WLCoref(model_name)

    else:
        raise Exception('Library "' + library + '" is not supported. You can choose one of: spacy, nltk, stanza and '
                                                'flair.')
    return model
