# Build from a patched copy so the upstream Git submodule stays clean.
find_package(PythonInterp 3 REQUIRED)
execute_process(
  COMMAND "${PYTHON_EXECUTABLE}"
    "${CMAKE_CURRENT_LIST_DIR}/../scripts/prepare_piqp.py"
    --source "${CMAKE_CURRENT_LIST_DIR}/../src/third_party/piqp"
    --destination "${CMAKE_CURRENT_BINARY_DIR}/care_piqp"
  RESULT_VARIABLE CARE_PIQP_PREPARE_STATUS
  OUTPUT_VARIABLE PIQP_INCLUDE_DIR
  ERROR_VARIABLE CARE_PIQP_PREPARE_ERROR
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT CARE_PIQP_PREPARE_STATUS EQUAL 0)
  message(FATAL_ERROR "Unable to prepare PIQP headers: ${CARE_PIQP_PREPARE_ERROR}")
endif()
