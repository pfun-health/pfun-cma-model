import shutil
import os

if __name__ == '__main__':
    shutil.move('./sdk', './sdk.zip')
    shutil.unpack_archive('./sdk.zip', './runtime/chalicelib/www')
    os.remove('./sdk.zip')
