cmake_minimum_required(VERSION 3.18)

if(APPLE AND NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET)
  if(DEFINED ENV{MACOSX_DEPLOYMENT_TARGET})
    set(
      CMAKE_OSX_DEPLOYMENT_TARGET "$ENV{MACOSX_DEPLOYMENT_TARGET}"
      CACHE STRING "macOS deployment target"
    )
  else()
    set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "macOS deployment target")
  endif()
endif()

project(neuralzip LANGUAGES C CXX)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE nanobind_ROOT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE NANOBIND_CMAKE_DIR_RESULT
)
if(NOT NANOBIND_CMAKE_DIR_RESULT EQUAL 0)
  message(FATAL_ERROR "Could not locate nanobind's CMake package.")
endif()
find_package(nanobind CONFIG REQUIRED)
find_package(Threads REQUIRED)

# Official Zstd revision pinned for reproducible, byte-compatible builds.
set(ZSTD_REVISION "5233c58e6ca0b1c4c6b353ad79649191ed195bdc")
set(ZSTD_ROOT "zstd-${ZSTD_REVISION}")
set(ZSTD_URL "https://codeload.github.com/facebook/zstd/tar.gz/${ZSTD_REVISION}")
set(ZSTD_SHA256 "4c8e7f2ab048339cf523e0ad62c119d4e4d49f4abf94028dffd5897706b1b68c")
set(ZSTD_ARCHIVE_ENV "NEURALZIP_ZSTD_ARCHIVE")

set(ZSTD_SOURCE_FILES
  lib/common/entropy_common.c
  lib/common/error_private.c
  lib/common/fse_decompress.c
  lib/compress/fse_compress.c
  lib/compress/hist.c
  lib/compress/huf_compress.c
  lib/decompress/huf_decompress.c
)

set(ZSTD_DEPENDENCY_DIR "${CMAKE_BINARY_DIR}/_deps")
file(MAKE_DIRECTORY "${ZSTD_DEPENDENCY_DIR}")

if(DEFINED ENV{${ZSTD_ARCHIVE_ENV}} AND NOT "$ENV{${ZSTD_ARCHIVE_ENV}}" STREQUAL "")
  set(ZSTD_ARCHIVE "$ENV{${ZSTD_ARCHIVE_ENV}}")
  get_filename_component(
    ZSTD_ARCHIVE "${ZSTD_ARCHIVE}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}"
  )
  if(NOT EXISTS "${ZSTD_ARCHIVE}" OR IS_DIRECTORY "${ZSTD_ARCHIVE}")
    message(FATAL_ERROR
      "${ZSTD_ARCHIVE_ENV} does not name a file: ${ZSTD_ARCHIVE}"
    )
  endif()
  set(ZSTD_ARCHIVE_IS_OVERRIDE TRUE)
else()
  set(ZSTD_ARCHIVE "${ZSTD_DEPENDENCY_DIR}/${ZSTD_ROOT}.tar.gz")
  set(ZSTD_ARCHIVE_IS_OVERRIDE FALSE)
  if(NOT EXISTS "${ZSTD_ARCHIVE}")
    file(
      DOWNLOAD "${ZSTD_URL}" "${ZSTD_ARCHIVE}"
      EXPECTED_HASH "SHA256=${ZSTD_SHA256}"
      STATUS ZSTD_DOWNLOAD_STATUS
      TIMEOUT 60
      TLS_VERIFY ON
    )
    list(GET ZSTD_DOWNLOAD_STATUS 0 ZSTD_DOWNLOAD_CODE)
    list(GET ZSTD_DOWNLOAD_STATUS 1 ZSTD_DOWNLOAD_MESSAGE)
    if(NOT ZSTD_DOWNLOAD_CODE EQUAL 0)
      file(REMOVE "${ZSTD_ARCHIVE}")
      message(FATAL_ERROR
        "Could not download pinned Zstd sources: ${ZSTD_DOWNLOAD_MESSAGE}. "
        "Set ${ZSTD_ARCHIVE_ENV} to the archive path for an offline build."
      )
    endif()
  endif()
endif()

file(SHA256 "${ZSTD_ARCHIVE}" ZSTD_ACTUAL_SHA256)
if(NOT ZSTD_ACTUAL_SHA256 STREQUAL ZSTD_SHA256)
  if(NOT ZSTD_ARCHIVE_IS_OVERRIDE)
    file(REMOVE "${ZSTD_ARCHIVE}")
  endif()
  message(FATAL_ERROR
    "Zstd archive SHA-256 mismatch: expected ${ZSTD_SHA256}, "
    "got ${ZSTD_ACTUAL_SHA256}."
  )
endif()

set(ZSTD_SOURCE_ROOT "${ZSTD_DEPENDENCY_DIR}/${ZSTD_ROOT}")
if(NOT EXISTS "${ZSTD_SOURCE_ROOT}/.neuralzip-complete")
  file(REMOVE_RECURSE "${ZSTD_SOURCE_ROOT}")
  file(
    ARCHIVE_EXTRACT
    INPUT "${ZSTD_ARCHIVE}"
    DESTINATION "${ZSTD_DEPENDENCY_DIR}"
  )
  file(TOUCH "${ZSTD_SOURCE_ROOT}/.neuralzip-complete")
endif()

set(ZSTD_SOURCES)
foreach(ZSTD_FILE IN LISTS ZSTD_SOURCE_FILES)
  list(APPEND ZSTD_SOURCES "${ZSTD_SOURCE_ROOT}/${ZSTD_FILE}")
endforeach()

nanobind_add_module(
  _huff0_cpp
  NB_STATIC
  NOMINSIZE
  neuralzip/_huff0_cpp.cpp
  ${ZSTD_SOURCES}
)
target_compile_features(_huff0_cpp PRIVATE cxx_std_17)
target_compile_definitions(_huff0_cpp PRIVATE ZSTD_DISABLE_ASM=1)
target_include_directories(
  _huff0_cpp
  PRIVATE
    "${ZSTD_SOURCE_ROOT}/lib"
    "${ZSTD_SOURCE_ROOT}/lib/common"
    "${ZSTD_SOURCE_ROOT}/lib/compress"
    "${ZSTD_SOURCE_ROOT}/lib/decompress"
)
target_link_libraries(_huff0_cpp PRIVATE Threads::Threads)
if(NOT MSVC)
  target_compile_options(_huff0_cpp PRIVATE -O3)
endif()

install(
  TARGETS _huff0_cpp
  LIBRARY DESTINATION neuralzip
  RUNTIME DESTINATION neuralzip
)

if(DEFINED SKBUILD_DATA_DIR AND NOT SKBUILD_DATA_DIR STREQUAL "")
  set(NEURALZIP_DATA_DIR "${SKBUILD_DATA_DIR}/share/neuralzip")
else()
  set(NEURALZIP_DATA_DIR "share/neuralzip")
endif()
install(
  FILES third_party/dfloat11/LICENSE
  DESTINATION "${NEURALZIP_DATA_DIR}/licenses/dfloat11"
)
install(
  FILES third_party/zstd/LICENSE third_party/zstd/UPSTREAM
  DESTINATION "${NEURALZIP_DATA_DIR}/licenses/zstd"
)
