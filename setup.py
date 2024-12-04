from setuptools import setup, find_packages

setup(
    name='project_name',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Flask',
        'torch',
        'transformers',
    ],
    author='Kaustubh Karanje',
    author_email='karanje.kaustubh23@gmail.com',
    description='Flask app with Hugging Face Transcriber-Medium model',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Kaustubh-Data-Scientist/Whisp_Speech_Recognize',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Framework :: Flask',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
