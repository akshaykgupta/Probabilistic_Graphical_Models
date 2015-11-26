#! /bin/bash
python addFeatures.py --file ner.traindev --use-cap TRUE --add-ing TRUE --add-at TRUE --add-hash TRUE --add-ly TRUE --add-nom-ver TRUE --use-url TRUE --add-poss TRUE --new-file prepared_ner_pos.traindev
./dmetaph prepared_ner_pos.traindev
mv prepared_ner_pos_phone.traindev prepared_ner_pos.traindev
java cc.mallet.fst.SimpleTagger --model-file model_pos_caps_ing_at_hash_ly_nv_url_poss_phone prepared_ner_pos.traindev > prepared_ner_pos_unlabelled.traindev
python postprocessing.py prepared_ner_caps.traindev prepared_ner_pos_unlabelled.traindev prepared_ner_caps_pos.traindev train