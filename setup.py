import setuptools

long_description = """\
'stratigraph' is a Python module for visualizing and analyzing stratigraphic models.
"""

setuptools.setup(
    name="stratigraph",
    version="0.0.1",
    author="Zoltan Sylvester",
    author_email="zoltan.sylvester@beg.utexas.edu",
    description="stratigraph: stratigraphic visualization and analysis",
    keywords = 'stratigraphy, chronostratigraphy, visualization',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zsylvester/stratigraph",
    packages=['stratigraph'],
    install_requires=['numpy','matplotlib',
        'scipy','mayavi','pillow','tqdm'],
    classifiers=[
        "Programming Language :: Python :: 3",
        'Intended Audience :: Science/Research',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)