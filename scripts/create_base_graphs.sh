# pixi run python scripts/create_maps_multi_scene.py graph.enable_loop_closure=false graph.node_culling_factor=1&
# pixi run python scripts/create_maps_multi_scene.py graph.enable_loop_closure=true graph.loop_closure_mode=oracle graph.node_culling_factor=1;
pixi run python scripts/create_maps_multi_scene.py graph.enable_loop_closure=true graph.loop_closure_mode=seqvlad graph.node_culling_factor=1;