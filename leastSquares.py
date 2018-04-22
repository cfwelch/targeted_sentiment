import numpy as np
import scipy
import scipy.sparse
from scipy import linalg
from numpy import dot
import NLU
import dicts
import entity_distance as ed
import math

smap = dict();
sset = set();
slist = list();
#list of class ids
classes = list();
#parallel list of class names
classNames = list();
#list of instructors
instructors = list();
#cache entities
entcache = dict();

def main():
    np.set_printoptions(threshold='nan');
    lines = NLU.getALines();
    utterances = NLU.getUtterances(lines);
    #do ALS stuff
    X = getMatrix(utterances)[0];
    idtestp = 28;
    print(X[idtestp]);
    cossim = consineUser(X, idtestp);
    classtest = 280;
    print(getWeightedGuess(cossim, X, classtest));

def getNearestNeighbor(cossim, X, entity):
    entID = -1;
    guess = 0;
    if is_number(entity):
        if str(entity) in classes:
            entID = classes.index(str(entity));
    elif entity in classNames:
        entID = classNames.index(entity);
    elif entity in instructors:
        entID = instructors.index(entity);
    if entID > 0:
        closest = -1;
        value = -1;
        for i in range(0, X.shape[0]):
            if cossim[i] > value:
                value = cossim[i];
                closest = X[i].tolist()[0][entID];
        guess = closest;
    else:
        guess = 5;
    return guess;

def getGuess(X, entity, speaker):
    entID = -1;
    guess = 0;
    if is_number(entity):
        if str(entity) in classes:
            entID = classes.index(str(entity));
    elif entity in classNames:
        entID = classNames.index(entity);
    elif entity in instructors:
        entID = instructors.index(entity);
    if entID > 0:
        guess = X[speaker].tolist()[0][entID];
    else:
        guess = 5;
    return guess;

def getWeightedGuess(cossim, X, entity):
    entID = -1;
    guess = 0;
    if is_number(entity):
        if str(entity) in classes:
            entID = classes.index(str(entity));
    elif entity in classNames:
        entID = classNames.index(entity);
    elif entity in instructors:
        entID = instructors.index(entity);
    if entID > 0:
        cshape = 0;
        for i in range(0, X.shape[0]):
            ################################################
            ###### USE FOR ALS
            guess += X[i].tolist()[0][entID] * cossim[i];
            cshape += 1;
            ###### USE FOR NON-ALS
            #if X[i].tolist()[entID] > -0.01:
            #    guess += X[i].tolist()[entID] * cossim[i];
            #    cshape += 1;
            ################################################
        if cshape > 0:
            guess /= cshape;
        else:
            guess = 5;
    else:
        guess = 5;
    return guess;

def consineUser(X, userID):
    consineWeights = list();
    for i in range(0, X.shape[0]):
        user1 = X[userID].tolist()[0];
        user2 = X[i].tolist()[0];
        num = np.dot(user1, user2);
        denom = linalg.norm(user1) * linalg.norm(user2);
        cosd = num/denom;
        consineWeights.append(cosd);
    return consineWeights;

def cosineUserWE(X, userID):
    consineWeights = list();
    for i in range(0, X.shape[0]):
        user1 = X[userID].tolist();
        user2 = X[i].tolist();
        dprod = 0.0;
        for j in range(0, len(user1)):
            if user1[j] != -1.0 and user2[j] != -1.0:
                dprod += user1[j]*user2[j];
        ss1 = 0.0;
        for j in range(0, len(user1)):
            if user1[j] != -1.0 and user2[j] != -1.0:
                ss1 += user1[j]*user1[j];
        ss1 = math.sqrt(ss1);
        ss2 = 0.0;
        for j in range(0, len(user2)):
            if user1[j] != -1.0 and user2[j] != -1.0:
                ss2 += user2[j]*user2[j];
        ss2 = math.sqrt(ss2);
        denom = ss1*ss2;
        if denom == 0:
            denom = 0.1;
        cosd = dprod/denom;
        consineWeights.append(cosd);
    return consineWeights;

def getMatrix(utterances):
    GROUNDTRUTHS = True;
    np.set_printoptions(threshold='nan');
    #lines = NLU.getALines();
    #do ALS stuff
    ioffset = len(classes);
    X = np.ones((len(sset), len(classes) + len(instructors))) * -1;
    #print(X.shape);
    for i in range(0, len(utterances)):
        slots = NLU.getSlots(utterances[i]);
        cslots = slots[0];
        islots = slots[1];
        for slot in islots:
            iname = "";
            if GROUNDTRUTHS:
                iname = slot[0];
            else:
                if slot[0] in entcache.keys():
                    iname = entcache[slot[0]];
                else:
                    iname = ed.entityDistance(slot[0])[1][1];
                    entcache[slot[0]] = iname;
            if slot[1] == "positive":
                X[slist.index(smap[utterances[i][0].strip()])][ioffset+instructors.index(iname)] = 10;
            elif slot[1] == "negative":
                X[slist.index(smap[utterances[i][0].strip()])][ioffset+instructors.index(iname)] = 0;
            elif slot[1] == "neutral":
                X[slist.index(smap[utterances[i][0].strip()])][ioffset+instructors.index(iname)] = 5;
        for slot in cslots:
            if is_number(slot[1]):
                if slot[1] in classes:
                    if slot[2] == "positive":
                        X[slist.index(smap[utterances[i][0].strip()])][classes.index(slot[1])] = 10;
                    elif slot[2] == "negative":
                        X[slist.index(smap[utterances[i][0].strip()])][classes.index(slot[1])] = 0;
                    elif slot[2] == "neutral":
                        X[slist.index(smap[utterances[i][0].strip()])][classes.index(slot[1])] = 5;
                else:
                    pass;
                    #print(slot[1] + " is not a class...");
            else:
                classname = "";
                if GROUNDTRUTHS:
                    classname = slot[1];
                else:
                    if slot[1] in entcache.keys():
                        classname = entcache[slot[1]];
                    else:
                        classname = ed.entityDistance(slot[1])[0][1];
                        entcache[slot[1]] = classname;
                if slot[2] == "positive":
                    X[slist.index(smap[utterances[i][0].strip()])][classNames.index(classname)] = 10;
                elif slot[2] == "negative":
                    X[slist.index(smap[utterances[i][0].strip()])][classNames.index(classname)] = 0;
                elif slot[2] == "neutral":
                    X[slist.index(smap[utterances[i][0].strip()])][classNames.index(classname)] = 5;
    # Add back these four lines and change return X to newX if you want to use ALS
    A,Y = nmf(X, 50);
    A = np.matrix(A);
    Y = np.matrix(Y);
    newX = A*Y;
    return newX, slist;

def is_number(s):
    try:
        int(s);
        return True;
    except ValueError:
        return False;

#e-6, 500iterations
def nmf(X, latent_features, max_iter=500, error_limit=1e-6, fit_error_limit=1e-6):
    """   Decompose X to A*Y   """
    eps = 1e-5
    #print 'Starting NMF decomposition with {} latent features and {} iterations.'.format(latent_features, max_iter)
    #X = X.toarray()  # I am passing in a scipy sparse matrix

    # mask
    mask = np.sign(X)
    mask[mask == -1] = 0;
    #X[X == -1] = 5;

    # initial matrices. A is random [0,1] and Y is A\X.
    rows, columns = X.shape
    A = np.random.rand(rows, latent_features)
    A = np.maximum(A, eps)

    Y = linalg.lstsq(A, X)[0]
    Y = np.maximum(Y, eps)

    masked_X = mask * X
    #print(masked_X);
    X_est_prev = dot(A, Y)
    for i in range(1, max_iter + 1):
        # ===== updates =====
        # Matlab: A=A.*(((W.*X)*Y')./((W.*(A*Y))*Y'));
        top = dot(masked_X, Y.T)
        bottom = (dot((mask * dot(A, Y)), Y.T)) + eps
        A *= top / bottom

        A = np.maximum(A, eps)
        # print 'A',  np.round(A, 2)

        # Matlab: Y=Y.*((A'*(W.*X))./(A'*(W.*(A*Y))));
        top = dot(A.T, masked_X)
        bottom = dot(A.T, mask * dot(A, Y)) + eps
        Y *= top / bottom
        Y = np.maximum(Y, eps)
        # print 'Y', np.round(Y, 2)

        # ==== evaluation ====
        if i % 5 == 0 or i == 1 or i == max_iter:
            #print 'Iteration {}:'.format(i),
            X_est = dot(A, Y)
            err = mask * (X_est_prev - X_est)
            fit_residual = np.sqrt(np.sum(err ** 2))
            X_est_prev = X_est

            curRes = linalg.norm(mask * (X - X_est), ord='fro')
            #print 'fit residual', np.round(fit_residual, 4),
            #print 'total residual', np.round(curRes, 4)
            if curRes < error_limit or fit_residual < fit_error_limit:
                break

    return A, Y

def onLoad():
    global classes;
    global instructors;
    classes = dicts.getEECSdict().keys();
    instructors = dicts.getEECSprofs();
    cdict = dicts.getEECSdict();
    for key in classes:
        classNames.append(cdict[key]);
    #read in the speaker file
    fi = open("../data/extract_samples/pID_AEU");
    slines = fi.readlines();
    fi.close();
    for i in range(0, len(slines)):
        parts = slines[i].strip().split("\t");
        smap[parts[1]] = parts[0];
        sset.add(parts[0]);
    slist2 = list(sset);
    for k in slist2:
        slist.append(k);

if __name__ != "__main__":
    onLoad();

if __name__ == "__main__":
    onLoad();
    main();
