classes = [
	'Albertsons',
	'BJs',
	'Costco',
	'CVSPharmacy',
	'FredMeyer',
	'Frys',
	'HarrisTeeter',
	'HEB',
	'HyVee',
	'JewelOsco',
	'KingSoopers',
	'Kroger',
	'Meijer',
	'Publix',
	'Safeway',
	'SamsClub',
	'ShopRite',
	'Smiths',
	'StopShop',
	'Target',
	'Walgreens',
	'Walmart',
	'Wegmans',
	'WholeFoodsMarket',
	'WinCoFoods'
	]

batch_size = 20
z_dim = 128

num_classes = len(classes)
validation_size = 0.2	
	
HEIGHT = 512
WIDTH = 256
IMAGE_DEPTH = 1

CROP_SIZE = (300, 300)
