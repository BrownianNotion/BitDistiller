# checkpoint 80
* PPL at 3919, might be a promising trajectory to look better than 1 bit,
though still quite bad and also the loss may even be higher
* PPL went back up to 5k at end of two epochs despite loss going down (this
was with wrong clipping, but realistically, not sure how much of an effect this would be)might need to just train to end and see...

# A40 train times
* Clipping ~9 min
* Train 2 epochs ~ 1hr