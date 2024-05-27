python run_infer.py \
--nr_types=0 \
--batch_size=64 \
--model_mode=fast \
--model_path=../model.pt \
tile \
--input_dir=../few/imgs/ \
--output_dir=../few/pred/ \
--mem_usage=0.1 \
--draw_dot \
--save_qupath
