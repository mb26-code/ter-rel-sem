
DROP TABLE IF EXISTS CORRESPONDANCE_PATRONS;
DROP TABLE IF EXISTS OCCURRENCES_PATRONS;
DROP TABLE IF EXISTS RELATIONS_CORPUS;

DROP TABLE IF EXISTS TYPES_RELATION;
DROP TABLE IF EXISTS PATRONS;

-----------------------------------------------------------------------------

CREATE TABLE TYPES_RELATION(
    id int PRIMARY KEY,
    nom text,
    description_ text
);

-----------------------------------------------------------------------------

-- pour mémoriser les patrons rencontrés lors du parcours du corpus
CREATE TABLE PATRONS(
  id SERIAL PRIMARY KEY,
  -- expression du patron lexico-sémantique
  prefixe text,  -- mot(s) avant l'entité 1
  type_entite1 text,  -- catégorie de l'entité 1 (ex: )
  infixe text,  -- mot(s) entre les entités
  type_entite2 text,  -- catégorie  de l'entité 2 (ex: )
  suffixe text,  -- mot(s) après l'entité 2

  expression_complete text NOT NULL -- concaténation des parties du patron
);

-----------------------------------------------------------------------------

CREATE TABLE RELATIONS_CORPUS (
    id SERIAL PRIMARY KEY,
    id_type_relation int,
    entite1 text,
    entite2 text,
    
    id_patron int,
    extrait text,

    reference text,
    
    CONSTRAINT rels_corpus_fk_id_patron
        FOREIGN KEY (id_patron) REFERENCES PATRONS(id),
    
    CONSTRAINT rels_corpus_fk_id_type_relation
        FOREIGN KEY (id_type_relation) REFERENCES TYPES_RELATION(id)
);

-----------------------------------------------------------------------------

CREATE TABLE OCCURRENCES_PATRONS(
    id_patron int,
    id_type_relation int,
    occurences int,

    CONSTRAINT occs_patrons_pk
        PRIMARY KEY (id_patron, id_type_relation),
    
    CONSTRAINT occs_patrons_fk_id_patron
        FOREIGN KEY (id_patron) REFERENCES PATRONS(id),
    
    CONSTRAINT occs_patrons_fk_id_type_relation
        FOREIGN KEY (id_type_relation) REFERENCES TYPES_RELATION(id)
);

-----------------------------------------------------------------------------

CREATE TABLE CORRESPONDANCE_PATRONS(
    id_patron int,
    id_type_relation int,
    score_fiabilite float,

    CONSTRAINT corr_patrons_pk
        PRIMARY KEY (id_patron, id_type_relation),
    
    CONSTRAINT corr_patrons_fk_id_patron
        FOREIGN KEY (id_patron) REFERENCES PATRONS(id),
    
    CONSTRAINT corr_patrons_fk_id_type_relation
        FOREIGN KEY (id_type_relation) REFERENCES TYPES_RELATION(id)
);

----------------------------------------------

SELECT * FROM TYPES_RELATION;
SELECT * FROM PATRONS;
SELECT * FROM RELATIONS_CORPUS;
SELECT * FROM OCCURRENCES_PATRONS;
SELECT * FROM CORRESPONDANCE_PATRONS;