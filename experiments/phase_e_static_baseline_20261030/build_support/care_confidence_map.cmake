# SHA checks used by the source-built ToF fusion and primitive geometry nodes.
find_package(OpenSSL REQUIRED)
target_link_libraries(tof_fusion_self_filter_node OpenSSL::Crypto)
target_link_libraries(body_sample_model OpenSSL::Crypto)
