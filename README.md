# FACE SICKNESS DETECTION
An approach to sickness diagnostic with computer vision

[![Python Version](https://img.shields.io/badge/python-3.7-brightgreen.svg)](https://python.org)
[![Django Version](https://img.shields.io/badge/django-2.1-brightgreen.svg)](https://djangoproject.com)



YOUTUBE AS YOUTUBE: [youtube.com/CollinceOkumu](https://youtu.be/0cNAVeTziIs) 

## Running the Project Locally
### setting up a virtual environment
```bash
sudo pip3 install virtualenv
```
```bash
virtualenv FaceSicknessDetection
```
```bash
source FaceSicknessDetection/bin/activate
```
## NOTE
Dont forget to download dlib facial landmark detection pretrained model.Include it on the root directory
[![Download Dlib landmark detector](http://dlib.net/dlib-logo.png)](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

First, clone the repository to your local machine:

```bash
git clone https://github.com/collinsokumu/FaceSicknessDetection.git
```

Install the requirements:

```bash
pip install -r requirements.txt
```

Apply the migrations:

```bash
python manage.py migrate
```

Finally, run the development server:

```bash
python manage.py runserver
```

The project will be available at **127.0.0.1:8000**.


## License

The source code is released under the [MIT License](https://github.com/sibtc/django-upload-example/blob/master/LICENSE).
## Gallary
![HEALTHY](https://github.com/collinsokumu/FaceSicknessDetection/blob/master/healthy1.png)
![SICK](https://github.com/collinsokumu/FaceSicknessDetection/blob/master/sick.png)
## References
[ILLNESS IS WRITTEN ALL OVER YOUR FACE](https://www.nature.com/articles/d41586-018-00125-2)

[Detecting Visually Observable Disease Symptoms from Face](https://www.researchgate.net/publication/307615602_Detecting_Visually_Observable_Disease_Symptoms_from_Faces)
## FUTURE WORK

- [ ] Train an end to end model tunned for sickness detection
- [ ] Enhance accuracy
- [ ] Determine the exact disease in patient through the same technology
- [ ] Work on the user interface
