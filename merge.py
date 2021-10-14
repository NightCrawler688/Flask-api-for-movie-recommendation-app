import csv
print('line2')
all_movies = []
headers = []
all_movie_links= []
print('line6')
with open("movies.csv") as f:
    csv_reader = csv.reader(f)
    data = list(csv_reader)
    all_movies = data[1:]
    headers = data[0]
headers.append('poster_link')
print('line13')
with open("final.csv",'a+') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(headers)
with open("movie_links.csv") as f:
        csv_reader = csv.reader(f)
        data = list(csv_reader)
        all_movie_links = data[1:]
for movie_item in  all_movies:
        poster_found  = any(movie_item[8] in movie_link_items for movie_link_items in all_movie_links)
        if poster_found: 
            for movie_link_item in all_movie_links:
                    if movie_item[8] == movie_link_item[0]:
                            movie_item.append(movie_link_item[1])
                            if len(movie_item) == 28:
                                    with open("final.csv",'a+') as f:
                                            csv_writer = csv.writer(f)
                                            csv_writer.writerow(movie_item)