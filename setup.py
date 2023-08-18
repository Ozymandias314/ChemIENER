from setuptools import setup, find_packages

setup( 
        name='ChemIENER',
        version='0.1.0',
        author='Vincent Fan', 
        author_email='vincentf@mit.edu',
        url='https://github.com/Ozymandias314/ChemIENER',
        packages=find_packages(),
        package_dir={'chemiener': 'chemiener'},
        python_requires='>=3.7',
        install_requires=[
            "numpy",
            "torch>=1.10.0,<2.0",
            "transformers>=4.6.0",
            "opencv-python==4.5.5.64",
            "opencv-python-headless==4.5.4.60",
            "Pillow==9.5.0",
            ],
        )
