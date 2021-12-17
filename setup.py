from setuptools import setup, find_packages

long_description = ""

setup(
    name = "keentune-brain",
    version = "1.0.0",
    description = "KeenTune brain unit",
    long_description = long_description,
    url = "https://gitee.com/anolis/keentune_brain",
    license = "MulanPSLv2",
    classifiers = [
        "Environment:: KeenTune",
        "IntendedAudience :: Information Technology",
        "IntendedAudience :: System Administrators",
        "License :: OSI Approved :: MulanPSLv2",
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.6",
        "ProgrammingLanguage :: Python"
    ],
    python_requires='>=3.6',
    packages = find_packages(),
    package_data={'brain': ['brain.conf']},
    
    data_files = [
        ("/etc/keentune/",["LICENSE"]),
        ("/etc/keentune/conf", ["brain/brain.conf"]),
    ],
)
