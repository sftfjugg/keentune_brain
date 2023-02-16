from setuptools import setup, find_packages

long_description = ""

setup(
    name        = "keentune-brain",
    version     = "2.0.1",
    description = "KeenTune brain unit",
    url         = "https://gitee.com/anolis/keentune_brain",
    license     = "MulanPSLv2",
    packages    = find_packages(),
    package_data= {'brain': ['brain.conf']},

    python_requires  = '>=3.6',
    long_description = long_description,

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
    data_files  = [
        ("/etc/keentune/brain", ["LICENSE"]),
        ("/etc/keentune/conf", ["brain/brain.conf"]),
    ],
    entry_points = {
        'console_scripts': ['keentune-brain=brain.brain:main']
    }
)
