#! /bin/bash
python addFeatures.py --file $1 --use-cap TRUE --add-ing TRUE --add-at TRUE --add-hash TRUE --add-ly TRUE --add-nom-ver TRUE --use-url TRUE --add-poss TRUE --new-file prepared_pos_caps_ing_at_hash_ly_nv_url_poss.test
./dmetaph prepared_pos_caps_ing_at_hash_ly_nv_url_poss.test
java cc.mallet.fst.SimpleTagger --model-file model_pos_caps_ing_at_hash_ly_nv_url_poss_phone prepared_pos_caps_ing_at_hash_ly_nv_url_poss_phone.test > prepared_pos_caps_ing_at_hash_ly_nv_url_poss_phone_unlabelled.test
python postprocessing.py prepared_pos_caps_ing_at_hash_ly_nv_url_poss_phone.test prepared_pos_caps_ing_at_hash_ly_nv_url_poss_phone_unlabelled.test $2
python addFeatures.py --file $1 --use-cap TRUE --new-file prepared_ner_caps.test
java cc.mallet.fst.SimpleTagger --model-file model_ner_caps prepared_ner_caps.test > prepared_ner_caps_unlabelled.test
python postprocessing.py prepared_ner_caps.test prepared_ner_caps_unlabelled.test $3