from glob import glob

from setuptools import find_packages, setup

package_name = "quadrucat_deploy"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/urdf", glob("urdf/*.urdf")),
        ("share/" + package_name + "/meshes", glob("meshes/*")),
        ("share/" + package_name + "/worlds", glob("worlds/*.sdf")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="Deploy quadrucat policies in Gazebo via ROS2",
    license="MIT",
    entry_points={
        "console_scripts": [
            "policy_node = quadrucat_deploy.policy_node:main",
        ],
    },
)
