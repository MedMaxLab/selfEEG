# license info

import os

import setuptools

requirements = [
    "pandas>=1.5.3",
    "scipy>=1.10.1",
    "torch>=2.0.0",
    "torchaudio>=2.0.2",
    "tqdm",
]

extra_require = {"interactive": ["jupyterlab", "scikit-learn", "seaborn", "mne"]}

PACKAGES = setuptools.find_packages(
    exclude=["test*", "Notebooks*", "docs*", "dist*", "dist_conda*"]
)

version_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "selfeeg", "VERSION.txt"))

with open(version_path, "r") as fd:
    version = fd.read().rstrip()

README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md")
with open(README_PATH) as readme_file:
    README = readme_file.read()

setuptools.setup(
    name="selfeeg",
    version=version,
    description="Self-Supervised Learning for EEG",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/MedMaxLab/selfEEG",
    author="MedMax Team",
    author_email="federico.delpup@studenti.unipd.it",
    packages=PACKAGES,
    license="MIT",
    license_files = ["LICENSE.md"],
    classifiers=[
        "Environment :: Console",
        "Environment :: GPU",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "Deep Learning",
        "Self-Supervised Learning",
        "Contrastive Learning",
        "Electroencephalography",
        "EEG",
    ],
    project_urls={
        "Bug Tracker": "https://github.com/MedMaxLab/selfEEG/issues",
        "Source Code": "https://github.com/MedMaxLab/selfEEG",
        "Documentation": "https://selfeeg.readthedocs.io/en/latest/index.html",
    },
    install_requires=requirements,
    include_package_data=True,
    python_requires=">=3.8",
    extras_require=extra_require,
    zip_safe=False,
)
