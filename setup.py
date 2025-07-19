from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="adup-routing-protocol",
    version="1.0.0",
    author="Nuwan Hapuarachchi",
    author_email="nuwanhapuarachchi@gmail.com",
    description="Advanced Distance-based Unified Protocol (ADUP) routing implementation with dynamic network simulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NuwanHapuarachchi/Routing-Protocol-ADUP-",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: System :: Networking",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    keywords="routing protocol, network simulation, ADUP, distance-vector, network protocols",
    project_urls={
        "Bug Reports": "https://github.com/NuwanHapuarachchi/Routing-Protocol-ADUP-/issues",
        "Source": "https://github.com/NuwanHapuarachchi/Routing-Protocol-ADUP-",
        "Documentation": "https://github.com/NuwanHapuarachchi/Routing-Protocol-ADUP-/blob/main/README.md",
    },
)
