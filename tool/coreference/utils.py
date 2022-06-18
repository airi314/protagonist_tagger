def load_model(library, model_name, save_singletons):
    if library == 'coreferee':
        from tool.coreference.coreferee import Coreferee
        if model_name is None:
            model_name = 'en_core_web_sm'
        model = Coreferee(model_name)

    elif library == 'fast_coref':
        from tool.coreference.fast_coref import FastCoref
        if model_name is None:
            model_name = 'ontonotes_best'
        model = FastCoref(model_name, save_singletons)

    elif library == 'long_doc_coref':
        from tool.coreference.long_doc_coref import LongDocCoref
        if model_name is None:
            model_name = 'ontonotes'
        model = LongDocCoref(model_name, save_singletons)

    elif library == 'wl_coref':
        from tool.coreference.wl_coref import WLCoref
        if model_name is None:
            model_name = "roberta"
        model = WLCoref(model_name, save_singletons)

    elif library == 'e2e_coref':
        from tool.coreference.e2e_coref import E2ECoref
        if model_name is None:
            model_name = "train_spanbert_large_ml0_d1/May10_03-28-49_54000"
        model = E2ECoref(model_name, save_singletons)

    elif library == 'neuralcoref':
        from tool.coreference.neuralcoref import NeuralCoref
        if model_name is None:
            model_name = "en_core_web_sm"
        model = NeuralCoref(model_name, save_singletons)

    else:
        raise Exception('Library "' + library + '" is not supported. You can choose one of: coreferee, fast_coref, long_doc_coref, wl_coref, '
                                                'e2e_coref and neuralcoref.')
    return model
