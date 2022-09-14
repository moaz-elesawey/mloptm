from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_desc = (here / 'README.md').read_text(encoding='utf-8')


setup(
        name='mloptm',
        version='1.0.0',
        description='Implementation of ML Optimization Methods in Python',
        long_description=long_desc,
        long_description_content_type='text/markdown',
        url='https://github.com/moaz-elesawey/mloptm',
        author='Moaz Mohammed El-Essawey',
        author_email='mohammedmiaz3141@gmail.com',
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3',
        ],
        keywords='python, python3, ml, optm, mloptm',
        package_dir={'': 'src'},
        packages=find_packages(where='src'),
        python_requires='>=3.7, <4',
        project_urls={
        'Source': 'https://github.com/moaz-elesawey/mloptm',
    },
)
