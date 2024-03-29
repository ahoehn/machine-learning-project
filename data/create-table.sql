CREATE TABLE income (
                        gemeinde_id INT PRIMARY KEY,
                        gemeinde_name VARCHAR,
                        median_income INT
);
CREATE TABLE gemeinde (
                          plz INT,
                          gemeinde_id INT,
                          gemeinde_name VARCHAR
);
