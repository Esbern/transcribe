UnboundLocalError                         Traceback (most recent call last)
Cell In[2], line 300, in _on_run(_)
    297 with out:
    298     audio_path = Path(w_audio.value)
--> 300     result = transcribe_window(
    301         audio_path=audio_path,
    302         target_start_s=float(w_start.value),
    303         target_duration_s=float(w_duration.value),
    304         left_context_s=float(w_left.value),
    305         right_context_s=float(w_right.value),
    306         chunk_length_s=float(w_chunk.value),
    307         stride_left_s=float(w_stride_l.value),
    308         stride_right_s=float(w_stride_r.value),
    309         num_beams=int(w_beams.value),
    310         condition_on_prev_tokens=bool(w_prev.value),
    311         no_speech_threshold=float(w_nospeech.value),
    312         repetition_penalty=float(w_reppen.value),
    313         denoise_strength=float(w_denoise.value),
    314         text_prompt=str(w_prompt.value),
    315         language=language,
    316     )
    318     print(f"Audio: {audio_path.name}")
    319     print(
    320         f"Target window:   {w_start.value:.2f} -> "
    321         f"{w_start.value + w_duration.value:.2f} s"
    322     )

Cell In[2], line 66, in transcribe_window(audio_path, target_start_s, target_duration_s, left_context_s, right_context_s, chunk_length_s, stride_left_s, stride_right_s, num_beams, condition_on_prev_tokens, no_speech_threshold, repetition_penalty, denoise_strength, text_prompt, language)
     63         prompt_ids = prompt_ids.to(device)
     64     generate_kwargs["prompt_ids"] = prompt_ids
---> 66 result = transcriber.pipe(
     67     {"array": y, "sampling_rate": sr},
     68     generate_kwargs=generate_kwargs,
     69 )
     71 local_target_start = target_start_s - expanded_start_s
     72 local_target_end = target_end_s - expanded_start_s

File c:\Users\holmes\.local\share\mamba\envs\transcribe\Lib\site-packages\transformers\pipelines\automatic_speech_recognition.py:275, in AutomaticSpeechRecognitionPipeline.__call__(self, inputs, **kwargs)
    218 def __call__(self, inputs: Union[np.ndarray, bytes, str, dict], **kwargs: Any) -> list[dict[str, Any]]:
    219     """
    220     Transcribe the audio sequence(s) given as inputs to text. See the [`AutomaticSpeechRecognitionPipeline`]
    221     documentation for more information.
   (...)    273                 `"".join(chunk["text"] for chunk in output["chunks"])`.
    274     """
--> 275     return super().__call__(inputs, **kwargs)

File c:\Users\holmes\.local\share\mamba\envs\transcribe\Lib\site-packages\transformers\pipelines\base.py:1459, in Pipeline.__call__(self, inputs, num_workers, batch_size, *args, **kwargs)
   1457     return self.iterate(inputs, preprocess_params, forward_params, postprocess_params)
   1458 elif self.framework == "pt" and isinstance(self, ChunkPipeline):
-> 1459     return next(
   1460         iter(
   1461             self.get_iterator(
   1462                 [inputs], num_workers, batch_size, preprocess_params, forward_params, postprocess_params
   1463             )
   1464         )
   1465     )
   1466 else:
   1467     return self.run_single(inputs, preprocess_params, forward_params, postprocess_params)

File c:\Users\holmes\.local\share\mamba\envs\transcribe\Lib\site-packages\transformers\pipelines\pt_utils.py:126, in PipelineIterator.__next__(self)
    123     return self.loader_batch_item()
    125 # We're out of items within a batch
--> 126 item = next(self.iterator)
    127 processed = self.infer(item, **self.params)
    128 # We now have a batch of "inferred things".

File c:\Users\holmes\.local\share\mamba\envs\transcribe\Lib\site-packages\transformers\pipelines\pt_utils.py:271, in PipelinePackIterator.__next__(self)
    268             return accumulator
    270 while not is_last:
--> 271     processed = self.infer(next(self.iterator), **self.params)
    272     if self.loader_batch_size is not None:
    273         if isinstance(processed, torch.Tensor):

File c:\Users\holmes\.local\share\mamba\envs\transcribe\Lib\site-packages\transformers\pipelines\base.py:1374, in Pipeline.forward(self, model_inputs, **forward_params)
   1372     with inference_context():
   1373         model_inputs = self._ensure_tensor_on_device(model_inputs, device=self.device)
-> 1374         model_outputs = self._forward(model_inputs, **forward_params)
   1375         model_outputs = self._ensure_tensor_on_device(model_outputs, device=torch.device("cpu"))
   1376 else:

File c:\Users\holmes\.local\share\mamba\envs\transcribe\Lib\site-packages\transformers\pipelines\automatic_speech_recognition.py:542, in AutomaticSpeechRecognitionPipeline._forward(self, model_inputs, return_timestamps, **generate_kwargs)
    536 main_input_name = self.model.main_input_name if hasattr(self.model, "main_input_name") else "inputs"
    537 generate_kwargs = {
    538     main_input_name: inputs,
    539     "attention_mask": attention_mask,
    540     **generate_kwargs,
    541 }
--> 542 tokens = self.model.generate(**generate_kwargs)
    544 # whisper longform generation stores timestamps in "segments"
    545 if return_timestamps == "word" and self.type == "seq2seq_whisper":

File c:\Users\holmes\.local\share\mamba\envs\transcribe\Lib\site-packages\transformers\models\whisper\generation_whisper.py:866, in WhisperGenerationMixin.generate(self, input_features, generation_config, logits_processor, stopping_criteria, prefix_allowed_tokens_fn, synced_gpus, return_timestamps, task, language, is_multilingual, prompt_ids, prompt_condition_type, condition_on_prev_tokens, temperature, compression_ratio_threshold, logprob_threshold, no_speech_threshold, num_segment_frames, attention_mask, time_precision, time_precision_features, return_token_timestamps, return_segments, return_dict_in_generate, force_unique_generate_call, monitor_progress, **kwargs)
    857             proc.set_begin_index(decoder_input_ids.shape[-1])
    859 # 6.6 Run generate with fallback
    860 (
    861     seek_sequences,
    862     seek_outputs,
    863     should_skip,
    864     do_condition_on_prev_tokens,
    865     model_output_type,
--> 866 ) = self.generate_with_fallback(
    867     segment_input=segment_input,
    868     decoder_input_ids=decoder_input_ids,
    869     cur_bsz=cur_bsz,
    870     seek=seek,
    871     batch_idx_map=batch_idx_map,
    872     temperatures=temperatures,
    873     generation_config=generation_config,
    874     logits_processor=logits_processor,
    875     stopping_criteria=stopping_criteria,
    876     prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    877     synced_gpus=synced_gpus,
    878     return_token_timestamps=return_token_timestamps,
    879     do_condition_on_prev_tokens=do_condition_on_prev_tokens,
    880     is_shortform=is_shortform,
    881     batch_size=batch_size,
    882     attention_mask=attention_mask,
    883     kwargs=kwargs,
    884 )
    886 # 6.7 In every generated sequence, split by timestamp tokens and extract segments
    887 for i, seek_sequence in enumerate(seek_sequences):

File c:\Users\holmes\.local\share\mamba\envs\transcribe\Lib\site-packages\transformers\models\whisper\generation_whisper.py:1085, in WhisperGenerationMixin.generate_with_fallback(self, segment_input, decoder_input_ids, cur_bsz, seek, batch_idx_map, temperatures, generation_config, logits_processor, stopping_criteria, prefix_allowed_tokens_fn, synced_gpus, return_token_timestamps, do_condition_on_prev_tokens, is_shortform, batch_size, attention_mask, kwargs)
   1082         seek_sequence = seek_sequence[:-num_paddings]
   1084 # check which sequences in batch need fallback & which should be skipped
-> 1085 needs_fallback[i], should_skip[i] = self._need_fallback(
   1086     seek_sequence,
   1087     seek_outputs,
   1088     i,
   1089     logits_processor,
   1090     generation_config,
   1091     self.config.vocab_size,
   1092     temperature,
   1093 )
   1095 # remove eos token
   1096 if seek_sequence[-1] == generation_config.eos_token_id:

File c:\Users\holmes\.local\share\mamba\envs\transcribe\Lib\site-packages\transformers\models\whisper\generation_whisper.py:1293, in WhisperGenerationMixin._need_fallback(self, seek_sequence, seek_outputs, index, logits_processor, generation_config, vocab_size, temperature)
   1287 if generation_config.no_speech_threshold is not None:
   1288     no_speech_prob = _get_attr_from_logit_processors(
   1289         logits_processor, WhisperNoSpeechDetection, "no_speech_prob"
   1290     )
   1292     if (
-> 1293         logprobs < generation_config.logprob_threshold
   1294         and no_speech_prob[index] > generation_config.no_speech_threshold
   1295     ):
   1296         needs_fallback = False
   1297         should_skip = True

UnboundLocalError: cannot access local variable 'logprobs' where it is not associated with a value