import {getTitleFromWikiDataID, groupInfoboxParams, addYearsInDict, writeDict, readJsonFile} from "./aux.mjs";
import wtf from "wtf_wikipedia";
import jsonData from './labelAnnotation/improved_questions.json' assert { type: "json" };
import path from "path";
import fs from "fs/promises";

function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
async function fetchInfoboxes(title, wikidataID) {
  let doc = await wtf.fetch(title);

  if(doc === null || doc.infoboxes().length === 0) {
    let title = await getTitleFromWikiDataID(wikidataID)
    doc = await wtf.fetch(title)
  }
  let officesTimeline = {}
  doc.infoboxes().forEach(infobox =>{
    let timeline = undefined
    if(infobox.type() === "football biography") {
      timeline = extractFootballBiography(infobox)
    }
    else if(infobox.type() === "cricketer") {
      timeline = extractCricketBiography(infobox)
    }
    // else if(infobox.type() === "basketball biography") {
    //   timeline = extractBasketballBiography(infobox)
    // }
    if(timeline !== undefined) {
      officesTimeline = {...officesTimeline, ...timeline};
    }
  });
  return officesTimeline;
}

function parseYears(timeline) {
  let startYear, endYear;
  if (timeline.includes("–")) {
    [startYear, endYear] = timeline.split("–");
    endYear = endYear === "" ? '2024' : endYear; // Default end year
  } else {
    startYear = timeline;
    endYear = undefined; // Undefined end year if not specified
  }
  return [startYear, endYear];
}

function extractFootballBiography(infobox) {
  let [groupedParams, maxFieldNr] = groupInfoboxParams(infobox, true)
  let clubTimeline = {}
  for (let key = 0; key <= maxFieldNr; key++ ) {
    let params = groupedParams[key]
    if(params === undefined)
        continue;
    // Handle club career
    if('clubs' in params && 'years' in params) {
      let team = params['clubs'].replace(/^→\s/, '');
      let [startYear, endYear] = parseYears(params['years']);
      addYearsInDict(clubTimeline, team, startYear, endYear);
    }
    if('youthyears' in params && 'youthclubs' in params) {
      let team = params['youthclubs'].replace(/^→\s/, '');
      let [startYear, endYear] = parseYears(params['youthyears']);
      addYearsInDict(clubTimeline, team, startYear, endYear);
    }

    // Handle international career
    if(params['nationalteam'] && params['nationalyears']) {
      let team = params['nationalteam'].replace(/^→\s/, '');
      let [startYear, endYear] = parseYears(params['nationalyears']);
      addYearsInDict(clubTimeline, team, startYear, endYear);
    }
  }

  return clubTimeline
}

function extractCricketBiography(infobox) {
  let [groupedParams, maxFieldNr] = groupInfoboxParams(infobox, true)
  let clubTimeline = {}
  for (let key = 0; key <= maxFieldNr; key++ ) {
    let params = groupedParams[key]
    if(params === undefined)
        continue;
    // Handle club career
    if('club' in params && 'year' in params) {
      let team = params['club'].replace(/^→\s/, '');
      let [startYear, endYear] = parseYears(params['year']);
      addYearsInDict(clubTimeline, team, startYear, endYear);
    }

    // Handle international career
    if( params['country'] && params['internationalspan']) {
      let team = params['country'] + " national cricket team";
      let [startYear, endYear] = parseYears(params['internationalspan']);
      addYearsInDict(clubTimeline, team, startYear, endYear);
    }
  }
  return clubTimeline;
}


// DOES NOT WORK -- library doesn't parse years right
function extractBasketballBiography(infobox) {
  let [groupedParams, maxFieldNr] = groupInfoboxParams(infobox, true)
  let clubTimeline = {}
  for (let key = 0; key <= maxFieldNr; key++ ) {
    let params = groupedParams[key]
    if(params === undefined)
        continue;
    // Handle club career
    if('team' in params && 'years' in params) {
      let team = params['team'];
      let [startYear, endYear] = parseYears(params['year']);
      addYearsInDict(clubTimeline, team, startYear, endYear);
    }
  }
  return clubTimeline;
}

async function fetchWithRetry(title, wikidataID, attempts = 3) {
  for (let i = 0; i < attempts; i++) {
    try {
      const result = await fetchInfoboxes(title, wikidataID);
      if (Object.keys(result).length > 0) {
        return result; // Return result if it's not empty
      }
    } catch (error) {
      console.error(`Attempt ${i + 1} failed for ${title}: ${error}`);
      await delay(1000 * (i + 1)); // Exponential back-off
    }
  }
  console.error(`All attempts failed for ${title}`);
  return {}; // Return empty if all attempts fail
}


async function processJsonData(jsonData) {
  let finalQuestions = [];
  for (const jsonOBJ of jsonData) {
    try {
      if (jsonOBJ['type'] === 'P54') {
        let timeline = await fetchInfoboxes(jsonOBJ['subject_label'],jsonOBJ['wikidata_ID']);
        let entry = {};
        entry['question'] = `List all teams  ${jsonOBJ['subject_label']} played to this day.`;
        entry['answers'] = timeline;
        entry['subject'] = jsonOBJ['subject']
        entry['wikidata_ID'] = jsonOBJ['wikidata_ID']
        entry['aliases'] = jsonOBJ['aliases']

        if(Object.keys(timeline).length !== 0)
          finalQuestions.push(entry);
        else
          continue;
        console.log('done with', jsonOBJ['subject_label']);
      }
    } catch (err) {
      // Log the error and the entity that caused it, then continue with the next entity
      console.error('Error processing', jsonOBJ['subject_label'], ':', err);
    }
  }
  return finalQuestions;
}

async function processJsonDataParallel(jsonData) {
  const promises = jsonData
    .filter(jsonOBJ => jsonOBJ['type'] === 'P54') // Assuming P54 is the relevant type
    .map(jsonOBJ => fetchWithRetry(jsonOBJ['subject_label'], jsonOBJ['wikidata_ID'])
      .then(timeline => {
        if (Object.keys(timeline).length === 0) return null; // Filter out empty timelines
        return {
          'question': `List all teams ${jsonOBJ['subject_label']} played for to this day.`,
          'answers': timeline
        };
      })
      .catch(err => {
        console.error('Error processing', jsonOBJ['subject_label'], ':', err);
        return null;
      })
    );

  const results = await Promise.all(promises);
  return results.filter(entry => entry !== null); // Filter out null entries resulting from errors or empty timelines
}

async function processInChunks(jsonData, chunkSize = 10, delayMs = 1000) {
  let results = [];

  for (let i = 0; i < jsonData.length; i += chunkSize) {
    // Process in chunks
    const chunk = jsonData.slice(i, i + chunkSize);
    const promises = chunk.map(jsonOBJ =>
      fetchWithRetry(jsonOBJ['subject_label'], jsonOBJ['wikidata_ID'])
        .then(timeline => {
          if (Object.keys(timeline).length === 0) return null;
          return {
            'question': `List all teams ${jsonOBJ['subject_label']} played for to this day.`,
            'answers': timeline
          };
        })
        .catch(err => {
          console.error('Error processing', jsonOBJ['subject_label'], ':', err);
          return null;
        })
    );

    const chunkResults = await Promise.allSettled(promises);
    // Extract successful results
    chunkResults.forEach(result => {
      if (result.status === 'fulfilled' && result.value !== null) {
        results.push(result.value);
      }
    });

    // Delay before next chunk to avoid hitting rate limits
    await delay(delayMs);
  }

  return results;
}


// fetchInfoboxes("Lionel Messi", "").then(result => {
//   console.log(result)
//   writeDict(result, "football.json")
// })
processJsonData(jsonData).then(finalQuestions => {
   writeDict(finalQuestions, "sports.json")
})

async function processSportPlayerFromFile(filepath) {
  let obj = await readJsonFile(filepath)
  let timeline = {}
  for(let i = 0; i <  obj['infoboxes_types'].length; i++) {
    let currentTimeline = undefined
    switch (obj['infoboxes_types'][i]) {
      case 'football biography':
        currentTimeline = extractFootballBiography(obj['infoboxes'][i])
        break
      case 'cricketer':
        currentTimeline = extractCricketBiography(obj['infoboxes'][i])
    }
    if(currentTimeline !== undefined) {
      timeline = {...timeline, ...currentTimeline};
    }
  }
  return timeline;
}

//const inputDirectory = './path/to/input/directory';
//const outputFile = './path/to/output/sports_timelines.json';

/**
 * Reads all file names in the directory and processes them in chunks.
 */
async function processAllFiles(inputDirectory, outputFile) {
    const files = await fs.readdir(inputDirectory);
    const txtFiles = files.filter(file => path.extname(file) === '.txt');
    const results = [];

    // Define batch size and process files in chunks
    const batchSize = 100;  // Adjust based on memory and performance needs
    for (let i = 0; i < txtFiles.length; i += batchSize) {
        const batchFiles = txtFiles.slice(i, Math.min(i + batchSize, txtFiles.length));
        const batchResults = await processBatch(batchFiles, inputDirectory);
        results.push(...batchResults);  // Collect results from each batch
    }

    // Write the collected results to a single output file
    writeDict(results, outputFile);
    console.log('All files have been processed and output written.');
}

/**
 * Processes a batch of files concurrently.
 * @param {array} batchFiles - Array of file names to process.
 * @param inputDirectory
 * @returns {Promise<array>} - A promise that resolves to an array of timeline results.
 */
async function processBatch(batchFiles, inputDirectory) {
    const promises = batchFiles.map(file => {
        const filePath = path.join(inputDirectory, file);
        return processSportPlayerFromFile(filePath).catch(err => {
            console.error(`Error processing file ${file}: ${err}`);
            return null;  // Continue with other files even if one fails
        });
    });
    return Promise.all(promises);
}


// let obj = await processSportPlayerFromFile("sportplayer.txt")
// console.log(obj)