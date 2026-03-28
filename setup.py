"""Setuptools packaging for PyroSense AI."""

from __future__ import annotations

from setuptools import find_packages, setup


setup(
    name="pyrosense-ai",
    version="1.0.0",
    description="Real-time Fire & Smoke Detection with Explainable Deep Learning",
    author="PyroSense AI Contributors",
    python_requires=">=3.10",
    packages=find_packages(exclude=("tests", "training.notebooks")),
    include_package_data=True,
    install_requires=[],
)

