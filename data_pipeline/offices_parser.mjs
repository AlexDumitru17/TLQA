import wtf from 'wtf_wikipedia';
import fs from 'fs';
import jsonData from './labelAnnotation/improved_questions.json' assert { type: "json" };
import fetch from 'node-fetch';
import wbkFn from 'wikibase-sdk';
const wbk = wbkFn({
  instance: 'https://www.wikidata.org',
  sparqlEndpoint: 'https://query.wikidata.org/sparql',
});
import {addYearsInDict, getTitleFromWikiDataID, groupInfoboxParams} from  "./aux.mjs";

function extractOfficeTimelineFromInfobox(infobox) {

  let [groupedParams, maxOfficeNr] = groupInfoboxParams(infobox)
  let officeTimeline = {};
  let lastOffice = null
  const normalizeKeys = (params) => {
    let normalizedParams = {};
    for (let key in params) {
      let normalizedKey = key.replace(/[^a-zA-Z]/g, '');
      normalizedParams[normalizedKey] = params[key];
    }
    return normalizedParams;
  };
  for (let key = 0; key <= maxOfficeNr; key++ ) {
    let params = groupedParams[key]
    if(params === undefined)
      continue;
    params = normalizeKeys(params); // Normalize the keys
    let startYear = null, endYear = null;
    let startYears = [], endYears = [];
    if ('termstart' in params) {
      const matches = params['termstart'].match(/\d{4}/);
      if (matches) startYear = parseInt(matches[0]);
    }
    if ('termend' in params) {
      const matches = params['termend'].match(/\d{4}/);
      if (matches) endYear = parseInt(matches[0]);
      else endYear = 2024; // Assuming current or future year if not specified
    }
    if('termstart' in params && !('termend' in params)) {
      endYear = 2024
    }
    let positionName = '';
       const positionTypes = ['office', 'jrsr', 'statesenate', 'stateassembly', 'suboffice', 'state', 'constituencymp', 'term', 'order', 'statedelegate', 'statehouse', 'statelegislature'];
    positionTypes.forEach(type => {
      if (params[type]) {
        switch (type) {
          case 'office':
            if (!params['suboffice']) {
              positionName = params[type];
            }
            break;
          case 'jrsr':
            positionName = `${params[type]} from ${params['state']}`;
            break;
          case 'statesenate':
            positionName = `Member of the ${params[type]} Senate`;
            break;
          case 'stateassembly':
            positionName = `Member of the ${params[type]} State Assembly`;
            break;
          case 'statehouse':
            positionName = `Member of the ${params[type]} House of Representatives`;
            break;
          case 'statedelegate':
            positionName = `Member of the ${params[type]} House of Delegates`;
            break;
          case 'statelegislature':
            positionName = `Member of the ${params[type]} Legislature`;
            break;
          case 'suboffice':
            positionName = params['office'] ? `${params['office']} for ${params[type]}` : params[type];
            let subterms = params['subterm'].split('–');
            startYear = subterms[0];
            endYear = subterms[1];
            break;
          case 'state':
            if (params['district']) {
              positionName = `Member of the U.S. House of Representatives from ${params[type]}`;
            }
            break;
          case 'constituencymp':
            positionName = `Member of Parliament for ${params[type]}`;
            break;
          case 'order':
            if (!params['office'] && !params['suboffice']) {
              positionName = params[type];
            }
            break;
          case 'term':
            let yearPeriods = params['term'].split('\n');
            for (let i = 0; i < yearPeriods.length; i++) {
              if (yearPeriods[i] === "") continue;
              let period = yearPeriods[i].split('–');
              startYear = period[0].match(/\d{4}/)[0];
              endYear = period[1].match(/\d{4}/)[0];
              startYears.push(startYear);
              endYears.push(endYear);
            }
            break;
        }
      }
    });
    if (positionName) {
          for(let i = 0; i < startYears.length - 1; i++)
            addYearsInDict(officeTimeline, positionName, startYears[i], endYears[i])
          addYearsInDict(officeTimeline, positionName, startYear, endYear);
          lastOffice = positionName;
        } else {
          if (lastOffice !== null) {
              for(let i = 0; i < startYears.length - 1; i++)
                addYearsInDict(officeTimeline, lastOffice, startYears[i], endYears[i])
              addYearsInDict(officeTimeline, lastOffice, startYear, endYear);
            }
        }
  }
  console.log(officeTimeline)
return officeTimeline;
}



async function fetchInfoboxes(title, wikidataID) {
  let doc = await wtf.fetch(title);
  if(doc === null || doc.infoboxes().length === 0) {
    let title = await getTitleFromWikiDataID(wikidataID)
    doc = await wtf.fetch(title)
  }
  let officesTimeline = {}
  doc.infoboxes().forEach(infobox =>{
    if(infobox.type() === "officeholder") {
      let timeline = extractOfficeTimelineFromInfobox(infobox)
      officesTimeline = {...officesTimeline, ...timeline};
    }
  });
  return officesTimeline;
}

async function processJsonData(jsonData) {
  let finalQuestions = [];
  for (const jsonOBJ of jsonData) {
    try {
      if (jsonOBJ['type'] === 'P39') {
        let timeline = await fetchInfoboxes(jsonOBJ['subject_label'],jsonOBJ['wikidata_ID']);
        let entry = {};
        entry['question'] = `List all political positions  ${jsonOBJ['subject_label']} held to this day.`;
        entry['answers'] = timeline;
        entry['subject'] = jsonOBJ['subject'];
        entry['wikidata_ID'] = jsonOBJ['wikidata_ID']
        entry['aliases']  = jsonOBJ['aliases']
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
    .filter(jsonOBJ => jsonOBJ['type'] === 'P39')
    .map(async jsonOBJ => {
      try {
        const timeline = await fetchInfoboxes(jsonOBJ['subject_label'], jsonOBJ['wikidata_ID']);
        const entry = {
          'question': `List all political positions ${jsonOBJ['subject_label']} held to this day.`,
          'answers': timeline,
          'subject': jsonOBJ['subject'],
          'wikidata_ID': jsonOBJ['wikidata_ID'],
          'aliases': jsonOBJ['aliases']
        };
        return entry;
      } catch (err) {
        // Log the error and the entity that caused it, then continue with the next entity
        console.error('Error processing', jsonOBJ['subject_label'], ':', err);
        return null;
      }
    });

  const results = await Promise.all(promises);
  const finalQuestions = results.filter(entry => entry !== null);
  return finalQuestions;
}

function main() {
  processJsonData(jsonData).then(finalQuestions => {
  const jsonString = JSON.stringify(finalQuestions, null, 2);
  try {
    fs.writeFileSync('output2.json', jsonString);
    console.log('JSON array has been written to output1.json');
  } catch (err) {
    console.error('Error writing file:', err);
  }
  });
}

// fetchInfoboxes("Winston Churchill", "Q1681029");
main()