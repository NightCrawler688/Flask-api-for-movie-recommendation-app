import csv
all_movies = []
with open('final.csv') as f:
        csv_reader = csv.reader(f)
        data = list(csv_reader)
        all_movies = data[1:]
liked_movies = []
disliked_movies = []
did_not_watch_movies = []
