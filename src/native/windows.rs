use windows::{
    core::{w, IUnknown, Interface, GUID, PCWSTR, PWSTR},
    Win32::System::{
        Com::{CoCreateInstance, CLSCTX_ALL, CLSCTX_INPROC_SERVER},
        Ole::{OleInitialize, OleUninitialize},
        Search::{
            CSearchManager, IAccessor, ICommand, ICommandText, IDBCreateCommand, IDBCreateSession,
            IDBInitialize, IDataInitialize, IRowset, ISearchCatalogManager, ISearchManager,
            ISearchQueryHelper, DBACCESSOR_ROWDATA, DBBINDING, DBMEMOWNER_CLIENTOWNED,
            DBPARAMIO_NOTPARAM, DBPART_VALUE, DBTYPE_WSTR, DB_NULL_HCHAPTER, HACCESSOR,
            MSDAINITIALIZE,
        },
    },
};

pub type SearchData = (String, String, String);
pub type SearchLocalFileResult<T> = Result<T, SearchLocalFileError>;

#[derive(Debug)]
pub enum SearchLocalFileError {
    #[allow(unused)]
    CreateAccessor(String),
    /// (Error Message, Cast to)
    #[allow(unused)]
    Cast(String, &'static str),
    /// (Error Message, column_index, row_index)
    #[allow(unused)]
    GetData(String, usize, usize),
    /// (Error Message, column_index, row_index)
    #[allow(unused)]
    ReleaseAccessor(String, usize, usize),
    /// (Error Message, column_index)
    #[allow(unused)]
    ReleaseRows(String, usize),
    /// (Error Message, Cast to)
    #[allow(unused)]
    CoCreateInstance(String, &'static str),
    #[allow(unused)]
    GetDataSource(String),
    #[allow(unused)]
    CreateSession(String),
    #[allow(unused)]
    CreateCommand(String),
    #[allow(unused)]
    DBInitialize(String),
    /// (Error Message, SQL Query)
    #[allow(unused)]
    SetCommandText(String, String),
    /// (Error Message, SQL Query)
    #[allow(unused)]
    Exeute(String, String),
    /// (SQL Query)
    #[allow(unused)]
    NotMatched(String),
    #[allow(unused)]
    GetCatalog(String),
    #[allow(unused)]
    GetQueryHelper(String),
    /// (Error Message, Set Query Kind)
    #[allow(unused)]
    SetQuery(String, &'static str),
    #[allow(unused)]
    GenerateSQLFromUserQuery(String),
    #[allow(unused)]
    OleInitialize(String),
}

/// `query` 문자열로 파일 시스템의 인덱스를 검색하는 함수
pub fn search_local_files_by_query(query: String) -> SearchLocalFileResult<Vec<SearchData>> {
    unsafe {
        OleInitialize(None).map_err(|e| SearchLocalFileError::OleInitialize(e.to_string()))?;
        let sql_query = create_search_query(query)?;
        let rowset = execute_search_query(sql_query)?;
        let result = process_query_data(&rowset)?;
        OleUninitialize();

        Ok(result)
    }
}

/// Default GUID for Search.CollatorDSO.1
const DBGUID_DEFAULT: GUID = GUID {
    data1: 0xc8b521fb,
    data2: 0x5cf3,
    data3: 0x11ce,
    data4: [0xad, 0xe5, 0x00, 0xaa, 0x00, 0x44, 0x77, 0x3d],
};

/// 쿼리 결과에 해당하는 데이터에 접근하기 위한 핸들을 생성하는 함수
unsafe fn create_accessor_handle(
    accessor: &IAccessor,
    index: usize,
) -> SearchLocalFileResult<HACCESSOR> {
    let bindings = DBBINDING {
        iOrdinal: index,
        obValue: 0,
        obStatus: 0,
        obLength: 0,
        dwPart: DBPART_VALUE.0 as u32,
        dwMemOwner: DBMEMOWNER_CLIENTOWNED.0 as u32,
        eParamIO: DBPARAMIO_NOTPARAM.0 as u32,
        cbMaxLen: 512,
        dwFlags: 0,
        wType: DBTYPE_WSTR.0 as u16,
        bPrecision: 0,
        bScale: 0,
        ..Default::default()
    };
    let mut status = 0;
    let mut accessor_handle = HACCESSOR::default();
    accessor
        .CreateAccessor(
            DBACCESSOR_ROWDATA.0 as u32,
            1,
            &bindings,
            0,
            &mut accessor_handle,
            Some(&mut status),
        )
        .map_err(|e| SearchLocalFileError::CreateAccessor(e.to_string()))?;

    Ok(accessor_handle)
}

/// Query로 나온 결과로부터 데이터를 추출하는 함수
unsafe fn process_query_data(rowset: &IRowset) -> SearchLocalFileResult<Vec<SearchData>> {
    let accessor: IAccessor = rowset
        .cast()
        .map_err(|e| SearchLocalFileError::Cast(e.to_string(), "IAccessor"))?;

    let mut output = Vec::new();
    let mut count = 0;
    loop {
        let mut rows_fetched = 0;
        let mut row_handles = [std::ptr::null_mut(); 1];
        let result = rowset.GetNextRows(
            DB_NULL_HCHAPTER as usize,
            0,
            &mut rows_fetched,
            &mut row_handles,
        );
        if result.is_err() {
            break;
        }
        if rows_fetched == 0 {
            break;
        }

        let mut data = Vec::new();

        for i in 0..3 {
            let mut item_name = [0u16; 512];

            let accessor_handle = create_accessor_handle(&accessor, i + 1)?;
            rowset
                .GetData(
                    *row_handles[0],
                    accessor_handle,
                    item_name.as_mut_ptr() as *mut _,
                )
                .map_err(|e| SearchLocalFileError::GetData(e.to_string(), count, i))?;
            let name = String::from_utf16_lossy(&item_name);
            // Remove null characters
            data.push(name.trim_end_matches('\u{0000}').to_string());

            accessor
                .ReleaseAccessor(accessor_handle, None)
                .map_err(|e| SearchLocalFileError::ReleaseAccessor(e.to_string(), count, i))?;
        }

        output.push((data[0].clone(), data[1].clone(), data[2].clone()));

        count += 1;
        rowset
            .ReleaseRows(
                1,
                row_handles[0],
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
            .map_err(|e| SearchLocalFileError::ReleaseRows(e.to_string(), count))?;
    }

    Ok(output)
}

/// OLEDB를 초기화 객체를 생성하는 함수
unsafe fn create_db_initialize() -> SearchLocalFileResult<IDBInitialize> {
    let data_init: IDataInitialize = CoCreateInstance(&MSDAINITIALIZE, None, CLSCTX_INPROC_SERVER)
        .map_err(|e| SearchLocalFileError::CoCreateInstance(e.to_string(), "IDataInitialize"))?;

    let mut unknown: Option<IUnknown> = None;
    data_init
        .GetDataSource(
            None,
            CLSCTX_INPROC_SERVER.0,
            w!("provider=Search.CollatorDSO.1;EXTENDED PROPERTIES=\"Application=Windows\""),
            &IDBInitialize::IID,
            &mut unknown as *mut _ as *mut _,
        )
        .map_err(|e| SearchLocalFileError::GetDataSource(e.to_string()))?;

    Ok(unknown
        .unwrap()
        .cast()
        .map_err(|e| SearchLocalFileError::Cast(e.to_string(), "IDBInitialize"))?)
}

/// SQL 쿼리를 실행하는 객체를 생성하는 함수
unsafe fn create_command(db_init: IDBInitialize) -> SearchLocalFileResult<ICommandText> {
    let db_create_session: IDBCreateSession = db_init
        .cast()
        .map_err(|e| SearchLocalFileError::Cast(e.to_string(), "IDBCreateSession"))?;
    let session: IUnknown = db_create_session
        .CreateSession(None, &IUnknown::IID)
        .map_err(|e| SearchLocalFileError::CreateSession(e.to_string()))?;
    let db_create_command: IDBCreateCommand = session
        .cast()
        .map_err(|e| SearchLocalFileError::Cast(e.to_string(), "IDBCreateCommand"))?;
    Ok(db_create_command
        .CreateCommand(None, &ICommand::IID)
        .map_err(|e| SearchLocalFileError::CreateCommand(e.to_string()))?
        .cast()
        .map_err(|e| SearchLocalFileError::Cast(e.to_string(), "ICommandText"))?)
}

/// SQL 쿼리를 실행하는 함수
unsafe fn execute_search_query(sql_query: PWSTR) -> SearchLocalFileResult<IRowset> {
    let db_init = create_db_initialize()?;
    db_init
        .Initialize()
        .map_err(|e| SearchLocalFileError::DBInitialize(e.to_string()))?;
    let command = create_command(db_init)?;

    // Set the command text
    command
        .SetCommandText(&DBGUID_DEFAULT, sql_query)
        .map_err(|e| {
            SearchLocalFileError::SetCommandText(e.to_string(), pwstr_to_string(sql_query))
        })?;

    // Execute the command
    let mut rowset: Option<IRowset> = None;
    command
        .Execute(
            None,
            &IRowset::IID,
            None,
            None,
            Some(&mut rowset as *mut _ as *mut _),
        )
        .map_err(|e| SearchLocalFileError::Exeute(e.to_string(), pwstr_to_string(sql_query)))?;
    rowset.ok_or(SearchLocalFileError::NotMatched(pwstr_to_string(sql_query)))
}

/// OLEDB가 이해할 수 있는 SQL 쿼리를 생성하는 함수
unsafe fn create_search_query(query_file_name: String) -> SearchLocalFileResult<PWSTR> {
    let search_manager: ISearchManager = CoCreateInstance(&CSearchManager, None, CLSCTX_ALL)
        .map_err(|e| SearchLocalFileError::CoCreateInstance(e.to_string(), "ISearchManager"))?;
    let catalog: ISearchCatalogManager = search_manager
        .GetCatalog(w!("SystemIndex"))
        .map_err(|e| SearchLocalFileError::GetCatalog(e.to_string()))?;
    let query_helper: ISearchQueryHelper = catalog
        .GetQueryHelper()
        .map_err(|e| SearchLocalFileError::GetQueryHelper(e.to_string()))?;

    // Windows Search API에서 접근 가능한 키워드 문서
    // https://learn.microsoft.com/ko-kr/windows/win32/properties/core-bumper
    query_helper
        .SetQuerySelectColumns(w!("System.FileName, System.ItemUrl, System.Size"))
        .map_err(|e| SearchLocalFileError::SetQuery(e.to_string(), "QuerySelectColumns"))?;
    query_helper
        .SetQueryWhereRestrictions(w!(
            "AND (System.FileExtension = '.pdf' OR System.FileExtension = '.png') AND System.Size < 10000000"
        ))
        .map_err(|e| SearchLocalFileError::SetQuery(e.to_string(), "QueryWhereRestrictions"))?;
    query_helper
        .SetQueryMaxResults(100)
        .map_err(|e| SearchLocalFileError::SetQuery(e.to_string(), "QueryMaxResults"))?;

    query_helper
        .GenerateSQLFromUserQuery(string_to_pcwstr(query_file_name))
        .map_err(|e| SearchLocalFileError::GenerateSQLFromUserQuery(e.to_string()))
}

fn string_to_pcwstr(str: String) -> PCWSTR {
    let mut v: Vec<u16> = str.encode_utf16().collect();
    v.push(0);
    PCWSTR::from_raw(v.as_ptr())
}

unsafe fn pwstr_to_string(pw_str: PWSTR) -> String {
    pw_str.to_string().unwrap_or(String::new())
}
