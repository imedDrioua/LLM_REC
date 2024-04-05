Search.setIndex({"docnames": ["API/LLM_augmentation", "API/data_loader", "API/decoder", "API/item_attributes", "API/loss_functions", "API/metrics", "API/mm_model", "API/model", "API/modules", "API/test", "API/train", "API/user_item_interactions", "API/user_profile", "index"], "filenames": ["API/LLM_augmentation.rst", "API/data_loader.rst", "API/decoder.rst", "API/item_attributes.rst", "API/loss_functions.rst", "API/metrics.rst", "API/mm_model.rst", "API/model.rst", "API/modules.rst", "API/test.rst", "API/train.rst", "API/user_item_interactions.rst", "API/user_profile.rst", "index.rst"], "titles": ["LLM augmentations", "Data loader module", "Decoder class module", "Item attribute augmentation prompt module", "Loss functions module", "Metrics module", "Multi model GCN module", "Model", "Data Loader", "Test class module", "Train class module", "User-Item interactions augmentation prompt module", "User profile augmentation prompt module", "Welcome to LLMrec\u2019s documentation!"], "terms": {"user": [0, 1, 2, 4, 5, 6, 9, 10], "item": [0, 2, 4, 5, 6, 9, 10], "interact": [0, 1, 6], "prompt": 0, "modul": [0, 7, 8, 13], "attribut": 0, "profil": 0, "thi": [1, 2, 6, 13], "defin": [1, 2, 3, 4, 5, 6, 9, 10, 11, 12], "class": [1, 6, 7, 13], "load": 1, "dataset": [1, 9, 10, 13], "provid": [1, 13], "some": 1, "util": 1, "function": [1, 2, 3, 6, 7, 11, 12], "sampl": [1, 5], "describ": 1, "For": 1, "each": 1, "new": 1, "should": [1, 2, 6], "creat": [1, 6], "implement": [1, 6], "follow": 1, "method": [1, 5], "__init__": 1, "initi": 1, "directori": 1, "batch": [1, 4, 9, 10], "size": [1, 4, 9, 10], "__len__": 1, "return": [1, 3, 4, 5, 6, 9, 10, 11, 12], "length": 1, "all": [1, 2, 5, 6], "dictionari": 1, "get_dataset": 1, "name": 1, "get_all_dataset": 1, "n_user": [1, 6], "from": [1, 6, 13], "train": [1, 6, 13], "posit": [1, 4, 5], "neg": [1, 4], "book": [1, 3, 11, 12], "print": 1, "shape": 1, "number": [1, 5, 10], "matrix": [1, 6], "sparsiti": 1, "data_load": 1, "booksdataset": 1, "data_dir": 1, "batch_siz": [1, 4, 9, 10], "1024": [1, 10], "base": [1, 2, 6, 9, 10], "object": [1, 9, 10], "none": [1, 6, 10], "type": [1, 4, 5, 9, 10, 11, 12], "dict": [1, 5, 9, 10], "paramet": [1, 3, 4, 5, 9, 10, 11, 12], "str": [1, 9, 11, 12], "numpi": 1, "ndarrai": 1, "int": [1, 5, 9], "list": [1, 5, 9, 11], "script": [2, 3, 4, 5, 9, 10, 11, 12], "feat_siz": 2, "embed_s": [2, 6], "64": 2, "forward": [2, 6], "comput": [2, 6], "perform": [2, 5, 6, 9], "everi": [2, 6], "call": [2, 6], "overridden": [2, 6], "subclass": [2, 6], "although": [2, 6], "recip": [2, 6], "pass": [2, 6], "need": [2, 6], "within": [2, 6], "one": [2, 6, 9], "instanc": [2, 6], "afterward": [2, 6], "instead": [2, 6], "sinc": [2, 6], "former": [2, 6], "take": [2, 6], "care": [2, 6], "run": [2, 6], "regist": [2, 6], "hook": [2, 6], "while": [2, 6], "latter": [2, 6], "silent": [2, 6], "ignor": [2, 6], "them": [2, 6], "data": [3, 6, 11, 12, 13], "item_attribut": 3, "llm_book_profil": 3, "llm": [3, 11, 12, 13], "inform": 3, "titl": 3, "year": 3, "author": 3, "model": [3, 4, 5, 9, 10, 11, 12, 13], "respons": [3, 11, 12], "loss_funct": 4, "bpr_loss_aug": 4, "pos_item": 4, "neg_item": 4, "prune_loss_drop_r": 4, "0": [4, 5, 6, 10], "71": 4, "decai": [4, 10], "1e": [4, 10], "05": [4, 10], "bayesian": 4, "person": 4, "rank": [4, 5], "bpr": 4, "embed": [4, 6, 9, 10], "drop": 4, "rate": [4, 5, 9], "prune": 4, "mf_loss": 4, "emb_loss": 4, "reg_loss": 4, "float": [4, 5, 10], "prune_loss": 4, "predict": [4, 5], "drop_rat": 4, "loss_upd": 4, "which": [5, 6, 9, 10], "ar": 5, "us": [5, 9, 10, 13], "evalu": [5, 10], "f1": 5, "pre": 5, "rec": 5, "calcul": [5, 10], "score": 5, "precis": 5, "recal": 5, "average_precis": 5, "r": 5, "cut": 5, "averag": 5, "relev": 5, "either": 5, "1": 5, "result": [5, 9, 10], "consid": 5, "calculate_auc": 5, "ground_truth": 5, "auc": 5, "ground": 5, "truth": 5, "dcg_at_k": 5, "k": [5, 9], "discount": 5, "cumul": 5, "gain": 5, "dcg": 5, "get_auc": 5, "item_scor": 5, "user_pos_test": 5, "get": 5, "area": 5, "under": 5, "curv": 5, "test": [5, 10, 13], "set": [5, 9, 10], "get_perform": 5, "valu": 5, "hit_at_k": 5, "hit": 5, "mean_average_precis": 5, "mean": 5, "ndcg_at_k": 5, "normal": 5, "precision_at_k": 5, "rank_list_by_heapq": 5, "test_item": 5, "heapq": 5, "rank_list_by_sort": 5, "sort": 5, "n": 5, "recall_at_k": 5, "all_pos_num": 5, "contain": 6, "mmmodel": 6, "i": [6, 9, 10, 13], "pytorch": 6, "mmgcn": 6, "mm_model": 6, "n_item": 6, "adjacency_matrix": 6, "image_embeddings_data": 6, "text_embeddings_data": 6, "book_attributes_data": 6, "user_profiles_data": 6, "n_layer": 6, "model_cat_r": 6, "02": 6, "train_df": 6, "arg": 6, "kwarg": 6, "create_adjacency_matrix": 6, "adjac": 6, "param": 6, "frame": 6, "user_indic": 6, "pos_item_indic": 6, "neg_item_indic": 6, "propag": 6, "user_embed": 6, "item_embed": 6, "rtype": 6, "torch": [6, 9, 10], "tensor": [6, 9, 10], "loader": 13, "tester": 9, "10": 9, "20": 9, "50": 9, "ua_embed": 9, "ia_embed": 9, "is_val": 9, "batch_test_flag": 9, "fals": 9, "bool": 9, "valid": 9, "flag": 9, "test_one_us": 9, "x": 9, "test_flag": 9, "part": 9, "u": 9, "": 9, "trainer": 10, "lr": 10, "0001": 10, "side_info_r": 10, "augmentation_r": 10, "012": 10, "test_us": 10, "feat_reg_loss_calcul": 10, "g_item_imag": 10, "g_item_text": 10, "g_user_imag": 10, "g_user_text": 10, "feat_reg_decai": 10, "featur": 10, "regular": 10, "loss": [7, 10], "imag": 10, "text": 10, "epoch": 10, "user_item_interact": 11, "llm_user_item_interact": 11, "user_histori": 11, "candid": 11, "histori": [11, 12], "read": [11, 12], "recommend": [11, 13], "user_profil": 12, "llm_user_profil": 12, "system": 13, "larg": 13, "languag": 13, "augment": 13, "collabor": 13, "filter": 13, "api": 13, "packag": 13, "we": 13, "prepar": 13, "bookscross": 13, "you": 13, "can": 13, "download": 13, "here": 13, "metric": 13, "index": 13, "search": 13, "page": 13, "multi": 7, "gcn": 7, "decod": 7}, "objects": {"": [[1, 0, 0, "-", "data_loader"], [2, 0, 0, "-", "decoder"], [3, 0, 0, "-", "item_attributes"], [4, 0, 0, "-", "loss_functions"], [5, 0, 0, "-", "metrics"], [6, 0, 0, "-", "mm_model"], [9, 0, 0, "-", "test"], [10, 0, 0, "-", "train"], [11, 0, 0, "-", "user_item_interactions"], [12, 0, 0, "-", "user_profile"]], "data_loader": [[1, 1, 1, "", "BooksDataset"]], "data_loader.BooksDataset": [[1, 2, 1, "", "describe"], [1, 2, 1, "", "get_all_datasets"], [1, 2, 1, "", "get_dataset"], [1, 2, 1, "", "sample"]], "decoder": [[2, 1, 1, "", "Decoder"]], "decoder.Decoder": [[2, 2, 1, "", "forward"]], "item_attributes": [[3, 3, 1, "", "llm_book_profile"]], "loss_functions": [[4, 3, 1, "", "bpr_loss_aug"], [4, 3, 1, "", "prune_loss"]], "metrics": [[5, 3, 1, "", "F1"], [5, 3, 1, "", "average_precision"], [5, 3, 1, "", "calculate_auc"], [5, 3, 1, "", "dcg_at_k"], [5, 3, 1, "", "get_auc"], [5, 3, 1, "", "get_performance"], [5, 3, 1, "", "hit_at_k"], [5, 3, 1, "", "mean_average_precision"], [5, 3, 1, "", "ndcg_at_k"], [5, 3, 1, "", "precision_at_k"], [5, 3, 1, "", "rank_list_by_heapq"], [5, 3, 1, "", "rank_list_by_sorted"], [5, 3, 1, "", "recall"], [5, 3, 1, "", "recall_at_k"]], "mm_model": [[6, 1, 1, "", "MmModel"]], "mm_model.MmModel": [[6, 2, 1, "", "create_adjacency_matrix"], [6, 2, 1, "", "forward"], [6, 2, 1, "", "propagate"]], "test": [[9, 1, 1, "", "Tester"]], "test.Tester": [[9, 2, 1, "", "test"], [9, 2, 1, "", "test_one_user"]], "train": [[10, 1, 1, "", "Trainer"]], "train.Trainer": [[10, 2, 1, "", "evaluate"], [10, 2, 1, "", "feat_reg_loss_calculation"], [10, 2, 1, "", "train"]], "user_item_interactions": [[11, 3, 1, "", "llm_user_item_interaction"]], "user_profile": [[12, 3, 1, "", "llm_user_profile"]]}, "objtypes": {"0": "py:module", "1": "py:class", "2": "py:method", "3": "py:function"}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "class", "Python class"], "2": ["py", "method", "Python method"], "3": ["py", "function", "Python function"]}, "titleterms": {"llm": 0, "augment": [0, 3, 11, 12], "data": [1, 8], "loader": [1, 8], "modul": [1, 2, 3, 4, 5, 6, 9, 10, 11, 12], "decod": 2, "class": [2, 9, 10], "item": [3, 11], "attribut": 3, "prompt": [3, 11, 12], "loss": 4, "function": 4, "metric": 5, "multi": 6, "model": [6, 7], "gcn": 6, "api": [], "test": 9, "train": 10, "user": [11, 12], "interact": 11, "profil": 12, "welcom": 13, "llmrec": 13, "": 13, "document": 13, "content": 13, "indic": 13, "tabl": 13}, "envversion": {"sphinx.domains.c": 3, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 9, "sphinx.domains.index": 1, "sphinx.domains.javascript": 3, "sphinx.domains.math": 2, "sphinx.domains.python": 4, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx": 60}, "alltitles": {"LLM augmentations": [[0, "llm-augmentations"]], "Data loader module": [[1, "data-loader-module"]], "Decoder class module": [[2, "module-decoder"]], "Item attribute augmentation prompt module": [[3, "module-item_attributes"]], "Loss functions module": [[4, "module-loss_functions"]], "Metrics module": [[5, "module-metrics"]], "Multi model GCN module": [[6, "module-mm_model"]], "Test class module": [[9, "module-test"]], "Train class module": [[10, "module-train"]], "User-Item interactions augmentation prompt module": [[11, "module-user_item_interactions"]], "User profile augmentation prompt module": [[12, "module-user_profile"]], "Welcome to LLMrec\u2019s documentation!": [[13, "welcome-to-llmrec-s-documentation"]], "Contents:": [[13, null]], "Indices and tables": [[13, "indices-and-tables"]], "Model": [[7, "model"]], "Data Loader": [[8, "data-loader"]]}, "indexentries": {}})