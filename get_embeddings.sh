python get_embeddings.py --model_name resnet18 --input_dir segment_embeddings_classification_dataset_norsz/ --output_dir embeddings_r18dml/ --fine_tune_ckpt dml_results/resnet18/best_model_state_dict_1.pth
python get_embeddings.py --model_name resnet50 --input_dir segment_embeddings_classification_dataset_norsz/ --output_dir embeddings_r50/
python get_embeddings.py --model_name resnet50 --input_dir segment_embeddings_classification_dataset_norsz/ --output_dir embeddings_r50dml/ --fine_tune_ckpt dml_results/resnet50/best_model_state_dict_1.pth
python get_embeddings.py --model_name resnet101 --input_dir segment_embeddings_classification_dataset_norsz/ --output_dir embeddings_r101dml/ --fine_tune_ckpt dml_results/resnet101/best_model_state_dict_1.pth
python get_embeddings.py --model_name resnet18 --input_dir segment_embeddings_classification_dataset_norsz/ --output_dir embeddings_r18/
python get_embeddings.py --model_name haralick --input_dir segment_embeddings_classification_dataset_norsz/ --output_dir embeddings_har/
