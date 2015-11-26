#! /bin/bash
echo "NER"
python addFeatures.py --file ner.test
java cc.mallet.fst.SimpleTagger --model-file model_ner prepared_ner.test > prepared_ner_unlabelled.test
python postprocessing.py prepared_ner.test prepared_ner_unlabelled.test prepared_ner_mylabel.test
python Fscore_ner.py prepared_ner_mylabel.test ner.test
echo "NER_CAPS"
python addFeatures.py --file ner.test --use-cap TRUE --new-file prepared_ner_caps.test
java cc.mallet.fst.SimpleTagger --model-file model_ner_caps prepared_ner_caps.test > prepared_ner_caps_unlabelled.test
python postprocessing.py prepared_ner_caps.test prepared_ner_caps_unlabelled.test prepared_ner_caps_mylabel.test
python Fscore_ner.py prepared_ner_caps_mylabel.test ner.test
echo "NER_CAPS_2"
python addFeatures.py --file ner.test --use-cap TRUE --new-file prepared_ner_caps.test
java cc.mallet.fst.SimpleTagger --model-file model_ner_caps_2 prepared_ner_caps.test > prepared_ner_caps_2_unlabelled.test
python postprocessing.py prepared_ner_caps.test prepared_ner_caps_2_unlabelled.test prepared_ner_caps_2_mylabel.test
python Fscore_ner.py prepared_ner_caps_2_mylabel.test ner.test
echo "NER_CAPS_POS"
python addFeatures.py --file ner.test --use-cap TRUE --add-ing TRUE --add-at TRUE --add-hash TRUE --add-ly TRUE --add-nom-ver TRUE --use-url TRUE --add-poss TRUE --new-file prepared_ner_pos.test
./dmetaph prepared_ner_pos.test
mv prepared_ner_pos_phone.test prepared_ner_pos.test
java cc.mallet.fst.SimpleTagger --model-file model_pos_caps_ing_at_hash_ly_nv_url_poss_phone prepared_ner_pos.test > prepared_ner_pos_unlabelled.test
python postprocessing.py prepared_ner_caps.test prepared_ner_pos_unlabelled.test prepared_ner_caps_pos.test test
java cc.mallet.fst.SimpleTagger --model-file model_ner_caps_pos prepared_ner_caps_pos.test > prepared_ner_caps_pos_unlabelled.test
python postprocessing.py prepared_ner_caps_pos.test prepared_ner_caps_pos_unlabelled.test prepared_ner_caps_pos_mylabel.test
python Fscore_ner.py prepared_ner_caps_pos_mylabel.test ner.test
echo "NER_CAPS_PLACE"
python placeFeatNer.py ner.test ner_place.test train
python addFeatures.py --file ner_place.test --use-cap TRUE --new-file prepared_ner_caps_place.test
java cc.mallet.fst.SimpleTagger --model-file model_ner_caps_place prepared_ner_caps_place.test > prepared_ner_caps_place_unlabelled.test
python postprocessing.py prepared_ner_caps_place.test prepared_ner_caps_place_unlabelled.test prepared_ner_caps_place_mylabel.test
python Fscore_ner.py prepared_ner_caps_place_mylabel.test ner.test