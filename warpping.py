import re  
import os  

def warppingCheck():
    with open(os.path.join(os.path.split(os.path.realpath(__file__))[0],"keentune-brain.spec"),'r') as f:
        spec = f.read()
        version_in_spec = re.search("Version:        ([\d.]+)\n",spec).group(1)
        release_in_spec = re.search("define anolis_release (\d)\n",spec).group(1)
        print("Get version: {}-{}".format(version_in_spec, release_in_spec))

        if re.search(" - {}-{}".format(version_in_spec, release_in_spec), spec):
            print("[OK] check the version of changelog at keentune-brain.spec.")
        else:
            print("[Failed] wrong version number in changelog at keentune-brain.spec.")
            return

    with open(os.path.join(os.path.split(os.path.realpath(__file__))[0],"setup.py"), 'r') as f:
        script = f.read()
        if re.search('version     = "{}",'.format(version_in_spec),script):
            print("[OK] check the version of setup.py.")
        else:
            print("[Failed] wrong version number in setup.py.")
            return

    print("Start wrap up of keentune-brain-{}-{}".format(version_in_spec, release_in_spec))
    return version_in_spec, release_in_spec

if __name__ == "__main__":
    version_in_spec, _ = warppingCheck()
    os.system("tar -cvzf keentune-brain-{}.tar.gz --exclude=**/__pycache__ brain keentune-brain.service LICENSE README.md requirements.txt setup.py".format(version_in_spec))