#! /bin/bash
echo "POS"
python addFeatures.py --file pos.test
java cc.mallet.fst.SimpleTagger --model-file model_pos prepared_pos.test > prepared_pos_unlabelled.test
python postprocessing.py prepared_pos.test prepared_pos_unlabelled.test prepared_pos_mylabel.test
python Fscore_pos.py prepared_pos_mylabel.test pos.test
echo "POS_CAPS"
python addFeatures.py --file pos.test --use-cap TRUE --new-file prepared_pos_caps.test
java cc.mallet.fst.SimpleTagger --model-file model_pos_caps prepared_pos_caps.test > prepared_pos_caps_unlabelled.test
python postprocessing.py prepared_pos_caps.test prepared_pos_caps_unlabelled.test prepared_pos_caps_mylabel.test
python Fscore_pos.py prepared_pos_caps_mylabel.test pos.test
echo "POS_CAPS_ING"
python addFeatures.py --file pos.test --use-cap TRUE --add-ing TRUE --new-file prepared_pos_caps_ing.test
java cc.mallet.fst.SimpleTagger --model-file model_pos_caps_ing prepared_pos_caps_ing.test > prepared_pos_caps_ing_unlabelled.test
python postprocessing.py prepared_pos_caps_ing.test prepared_pos_caps_ing_unlabelled.test prepared_pos_caps_ing_mylabel.test
python Fscore_pos.py prepared_pos_caps_ing_mylabel.test pos.test
echo "POS_CAPS_ING_AT"
python addFeatures.py --file pos.test --use-cap TRUE --add-ing TRUE --add-at TRUE --new-file prepared_pos_caps_ing_at.test
java cc.mallet.fst.SimpleTagger --model-file model_pos_caps_ing_at prepared_pos_caps_ing_at.test > prepared_pos_caps_ing_at_unlabelled.test
python postprocessing.py prepared_pos_caps_ing_at.test prepared_pos_caps_ing_at_unlabelled.test prepared_pos_caps_ing_at_mylabel.test
python Fscore_pos.py prepared_pos_caps_ing_at_mylabel.test pos.test
echo "POS_CAPS_ING_AT_HASH"
python addFeatures.py --file pos.test --use-cap TRUE --add-ing TRUE --add-at TRUE --add-hash TRUE --new-file prepared_pos_caps_ing_at_hash.test
java cc.mallet.fst.SimpleTagger --model-file model_pos_caps_ing_at_hash prepared_pos_caps_ing_at_hash.test > prepared_pos_caps_ing_at_hash_unlabelled.test
python postprocessing.py prepared_pos_caps_ing_at_hash.test prepared_pos_caps_ing_at_hash_unlabelled.test prepared_pos_caps_ing_at_hash_mylabel.test
python Fscore_pos.py prepared_pos_caps_ing_at_hash_mylabel.test pos.test
echo "POS_CAPS_ING_AT_HASH_LY"
python addFeatures.py --file pos.test --use-cap TRUE --add-ing TRUE --add-at TRUE --add-hash TRUE --add-ly TRUE --new-file prepared_pos_caps_ing_at_hash_ly.test
java cc.mallet.fst.SimpleTagger --model-file model_pos_caps_ing_at_hash_ly prepared_pos_caps_ing_at_hash_ly.test > prepared_pos_caps_ing_at_hash_ly_unlabelled.test
python postprocessing.py prepared_pos_caps_ing_at_hash_ly.test prepared_pos_caps_ing_at_hash_ly_unlabelled.test prepared_pos_caps_ing_at_hash_ly_mylabel.test
python Fscore_pos.py prepared_pos_caps_ing_at_hash_ly_mylabel.test pos.test
echo "POS_CAPS_ING_AT_HASH_LY_NV"
python addFeatures.py --file pos.test --use-cap TRUE --add-ing TRUE --add-at TRUE --add-hash TRUE --add-ly TRUE --add-nom-ver TRUE --new-file prepared_pos_caps_ing_at_hash_ly_nv.test
java cc.mallet.fst.SimpleTagger --model-file model_pos_caps_ing_at_hash_ly_nv prepared_pos_caps_ing_at_hash_ly_nv.test > prepared_pos_caps_ing_at_hash_ly_nv_unlabelled.test
python postprocessing.py prepared_pos_caps_ing_at_hash_ly_nv.test prepared_pos_caps_ing_at_hash_ly_nv_unlabelled.test prepared_pos_caps_ing_at_hash_ly_nv_mylabel.test
python Fscore_pos.py prepared_pos_caps_ing_at_hash_ly_nv_mylabel.test pos.test
echo "POS_CAPS_ING_AT_HASH_LY_NV_URL"
python addFeatures.py --file pos.test --use-cap TRUE --add-ing TRUE --add-at TRUE --add-hash TRUE --add-ly TRUE --add-nom-ver TRUE --use-url TRUE --new-file prepared_pos_caps_ing_at_hash_ly_nv_url.test
java cc.mallet.fst.SimpleTagger --model-file model_pos_caps_ing_at_hash_ly_nv_url prepared_pos_caps_ing_at_hash_ly_nv_url.test > prepared_pos_caps_ing_at_hash_ly_nv_url_unlabelled.test
python postprocessing.py prepared_pos_caps_ing_at_hash_ly_nv_url.test prepared_pos_caps_ing_at_hash_ly_nv_url_unlabelled.test prepared_pos_caps_ing_at_hash_ly_nv_url_mylabel.test
python Fscore_pos.py prepared_pos_caps_ing_at_hash_ly_nv_url_mylabel.test pos.test
echo "POS_CAPS_ING_AT_HASH_LY_NV_URL_POSS"
python addFeatures.py --file pos.test --use-cap TRUE --add-ing TRUE --add-at TRUE --add-hash TRUE --add-ly TRUE --add-nom-ver TRUE --use-url TRUE --add-poss TRUE --new-file prepared_pos_caps_ing_at_hash_ly_nv_url_poss.test
java cc.mallet.fst.SimpleTagger --model-file model_pos_caps_ing_at_hash_ly_nv_url_poss prepared_pos_caps_ing_at_hash_ly_nv_url_poss.test > prepared_pos_caps_ing_at_hash_ly_nv_url_poss_unlabelled.test
python postprocessing.py prepared_pos_caps_ing_at_hash_ly_nv_url_poss.test prepared_pos_caps_ing_at_hash_ly_nv_url_poss_unlabelled.test prepared_pos_caps_ing_at_hash_ly_nv_url_poss_mylabel.test
python Fscore_pos.py prepared_pos_caps_ing_at_hash_ly_nv_url_poss_mylabel.test pos.test
echo "POS_CAPS_ING_AT_HASH_LY_NV_URL_POSS_PHONE"
java cc.mallet.fst.SimpleTagger --model-file model_pos_caps_ing_at_hash_ly_nv_url_poss_phone prepared_pos_caps_ing_at_hash_ly_nv_url_poss_phone.test > prepared_pos_caps_ing_at_hash_ly_nv_url_poss_phone_unlabelled.test
python postprocessing.py prepared_pos_caps_ing_at_hash_ly_nv_url_poss_phone.test prepared_pos_caps_ing_at_hash_ly_nv_url_poss_phone_unlabelled.test prepared_pos_caps_ing_at_hash_ly_nv_url_poss_phone_mylabel.test
python Fscore_pos.py prepared_pos_caps_ing_at_hash_ly_nv_url_poss_phone_mylabel.test pos.test

