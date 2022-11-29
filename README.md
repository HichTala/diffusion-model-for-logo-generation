# Diffusion Model For Logo Generation
This provides a functional scraping tool that enables you to get a labellized logo dataset (~500) from [SHL website]('https://sportslogohistory.com/').

### How to

Just edit the `targets.txt` file with adding the different page urls you want to scrap.
Separate them with line wrap. The given `targets.txt` file is used to scrap the whole website.

To get your dataset, then run :

```bash
    python3.10 datagetter.py
```

You will get a `dataset.csv` file containing image url and description plus a `dataset.tar` file containing logos and captions.