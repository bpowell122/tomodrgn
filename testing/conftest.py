import glob
import os
import shutil

import pytest


@pytest.fixture(scope='session')
def output_dir(tmp_path_factory):
    # Create a temporary directory for all outputs stored at tomodrgn_source_dir/testing/output
    # output_dir = files('tomodrgn').parent / 'testing' / 'output'
    output_dir = tmp_path_factory.getbasetemp()
    print(f'{output_dir=}')

    # delete any pre-existing contents of this dir
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # make the output dir
    os.mkdir(output_dir)

    # Yield the directory path to the tests
    yield output_dir

    # Cleanup the directory after all tests (byproduct of pytest-console-scripts, normally not noticed because pytest defaults to output in tmp dir
    # dirs_to_remove = glob.glob(f'{output_dir}/*subprocess*') + glob.glob(f'{output_dir}/*inprocess*')
    # for dir_to_remove in dirs_to_remove:
    #     try:
    #         # remove symlinks
    #         os.unlink(dir_to_remove)
    #     except IsADirectoryError:
    #         # remove directories
    #         shutil.rmtree(dir_to_remove)
    dirs_to_remove = glob.glob(f'{output_dir}/test_*')
    for dir_to_remove in dirs_to_remove:
        # remove symlinks and associated dirs
        if os.path.islink(dir_to_remove):
            os.unlink(dir_to_remove)
        elif os.path.isdir(dir_to_remove):
            shutil.rmtree(dir_to_remove)
        else:
            # do not remove potential globbed test_* files, not that any are expected to exist in this dir
            pass

