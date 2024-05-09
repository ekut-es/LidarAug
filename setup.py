import os
from setuptools import setup
from setuptools.command.install import install

MODULE_NAME = "LidarAug"

SO_FILES = [
    "./cpp/build_files/src/transformations/transformations.cpython-312-x86_64-linux-gnu.so",
    "./cpp/build_files/src/weather_simulations/weather_simulations.cpython-312-x86_64-linux-gnu.so",
]


class CustomInstall(install):

    def run(self):
        install.run(self)
        self.copy_so_files()

    def copy_so_files(self):
        print("Starting to copy shared object files...")
        for so_file in SO_FILES:

            dest_dir = os.path.join(self.install_lib, MODULE_NAME)
            os.makedirs(dest_dir, exist_ok=True)

            print(f"copying {so_file} -> {dest_dir}")
            self.copy_file(so_file, dest_dir)


setup(cmdclass={"install": CustomInstall}, )
