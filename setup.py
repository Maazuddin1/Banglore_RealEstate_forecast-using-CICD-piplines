from setuptools import setup, find_packages

setup(
    name="Banglore_house_price_estimator",
    version="1.0",
    description="A machine learning project for house price prediction in Banglore",
    author="Maaz uddin",
    packages=find_packages(),
    install_requires=[
        "flask",
        "pandas",
        "numpy",
        "scikit-learn",
        "seaborn",
        "matplotlib"
    ]
)
