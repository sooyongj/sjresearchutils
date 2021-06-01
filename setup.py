from setuptools import setup, find_packages

setup(name='sjresearchutil',
      version='0.1.0',
      url='',
      license='MIT',
      author='Sooyong Jang',
      author_email='sooyong@seas.upenn.edu',
      packages=find_packages(exclude=['tests']),
      long_description=open('README.md').read(),
      zip_safe=False,
      setup_requires=[''],
      test_suite='nose.collector')
