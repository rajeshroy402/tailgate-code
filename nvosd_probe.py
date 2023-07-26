# 4.3
# Define the Probe Function
def osd_sink_pad_buffer_probe(pad, info, u_data):
    gst_buffer=info.get_buffer()

    # Retrieve batch metadata from the gst_buffer
    batch_meta=pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame=batch_meta.frame_meta_list
    while l_frame is not None:
        
        # Initially set the tailgate indicator to False for each frame
        tailgate=False
        try:
            frame_meta=pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        frame_number=frame_meta.frame_num
        l_obj=frame_meta.obj_meta_list
        
        # Iterate through each object to check its dimension
        while l_obj is not None:
            try:
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
                
                # If the object meet the criteria then set tailgate indicator to True
                obj_bottom=obj_meta.rect_params.top+obj_meta.rect_params.height
                if (obj_meta.rect_params.width > FRAME_WIDTH*.3) & (obj_bottom > FRAME_HEIGHT*.9): 
                    tailgate=True
                    
            except StopIteration:
                break
            try: 
                l_obj=l_obj.next
            except StopIteration:
                break
                
        display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        # Setting display text to be shown on screen
        # Note that the pyds module allocates a buffer for the string, and the
        # memory will not be claimed by the garbage collector.
        # Reading the display_text field here will return the C address of the
        # allocated string. Use pyds.get_string() to get the string content.
        py_nvosd_text_params.display_text = "Frame Number={} Tailgate={}".format(frame_number, tailgate)

        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12

        # Font , font-color and font-size
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 36
        # set(red, green, blue, alpha); set to White
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        # Text background color
        py_nvosd_text_params.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        # Using pyds.get_string() to get display_text as string
        print(pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        print(f'Analyzing frame {frame_number}', end='\r')
        inference_output.append(str(int(tailgate)))
        try:
            l_frame=l_frame.next
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK