-- Table : Continent
CREATE TABLE Continent (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(255) NOT NULL
);

-- Table : Country
CREATE TABLE Country (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(255) NOT NULL,
    code3 VARCHAR(255),
    population BIGINT,
    continent_id INT,
    FOREIGN KEY (continent_id) REFERENCES Continent(id)
);

-- Table : Region
CREATE TABLE Region (
    id INT PRIMARY KEY AUTO_INCREMENT,
    country_id INT,
    name VARCHAR(255),
    FOREIGN KEY (country_id) REFERENCES Country(id)
);

-- Table : Disease
CREATE TABLE Disease (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(255) NOT NULL
);

-- Table : Global_Data
CREATE TABLE Global_Data (
    id INT PRIMARY KEY AUTO_INCREMENT,
    country_id INT,
    disease_id INT,
    date DATE,
    total_cases INT,
    new_cases INT,
    total_deaths INT,
    new_deaths INT,
    total_recovered INT,
    new_recovered INT,
    active_cases INT,
    serious_critical INT,
    total_tests BIGINT,
    tests_per_million INT,
    FOREIGN KEY (country_id) REFERENCES Country(id),
    FOREIGN KEY (disease_id) REFERENCES Disease(id)
);