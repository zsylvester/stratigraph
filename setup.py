import setuptools

long_description = """\
'stratigraph' is a Python package for stratigraphic visualization and analysis
"""

setuptools.setup(
    name="stratigraph",
    version="0.1.2",
    author="Zoltan Sylvester",
    author_email="zoltan.sylvester@beg.utexas.edu",
    description="A Python module for stratigraphic visualization and analysis",
    keywords = "geology, geoscience, stratigraphy, chronostratigraphy, visualization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zsylvester/stratigraph",
    packages=['stratigraph'],
    install_requires=["numpy", "scipy", "scikit-learn", "scikit-image", 
        "shapely", "pillow", "matplotlib", "mayavi", "tqdm"],
    classifiers=[
        "Programming Language :: Python :: 3",
        'Intended Audience :: Science/Research',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
